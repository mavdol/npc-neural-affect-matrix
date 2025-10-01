use ort::{
    session::{
        Session,
        builder::GraphOptimizationLevel,
    },
    value::Value,
    inputs,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;
use thiserror::Error;
use ndarray::Array2;

#[derive(Error, Debug, Clone)]
pub enum EmotionPredictorError {
    #[error("IO error: {0}")]
    Io(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Model loading error: {0}")]
    ModelLoading(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("ONNX Runtime error: {0}")]
    OnnxRuntime(String),

    #[error("Array shape error: {0}")]
    ArrayShape(String),

}

impl From<std::io::Error> for EmotionPredictorError {
    fn from(error: std::io::Error) -> Self {
        EmotionPredictorError::Io(error.to_string())
    }
}

impl From<ort::Error> for EmotionPredictorError {
    fn from(error: ort::Error) -> Self {
        EmotionPredictorError::OnnxRuntime(error.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionPrediction {
    pub valence: f32,
    pub arousal: f32,
}

impl EmotionPrediction {
    pub fn new(valence: f32, arousal: f32) -> Self {
        Self { valence, arousal }
    }

    pub fn values(&self) -> (f32, f32) {
        (self.valence, self.arousal)
    }
}

pub struct EmotionPredictor {
    session: Session,
    tokenizer: Tokenizer,
    max_length: usize,
}

impl EmotionPredictor {
    const MODEL_VERSION: &'static str = "v0.0.1";

    pub fn new() -> Result<Self, EmotionPredictorError> {
        ort::init()
            .with_name("emotion_prediction")
            .commit()?;

        let cache_dir = Self::get_cache_directory()?;
        let model_dir = cache_dir.join(format!("NPC-Prediction-Model-{}", Self::MODEL_VERSION));

        Self::check_and_download_models()?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Self::load_tokenizer_with_fallback(&tokenizer_path)?;

        let onnx_model_path = model_dir.join("model.onnx");
        let model_data = std::fs::read(&onnx_model_path)?;
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_memory(&model_data)?;

        Ok(Self {
            session,
            tokenizer,
            max_length: 512,
        })
    }

    pub fn check_and_download_models() -> Result<String, EmotionPredictorError> {
        let cache_dir = Self::get_cache_directory()?;
        let model_dir = cache_dir.join(format!("NPC-Prediction-Model-{}", Self::MODEL_VERSION));

        if Self::models_exist_and_valid(&model_dir) {
            return Ok("Models already up to date".to_string());
        }

        Self::cleanup_old_versions(&cache_dir)?;

        Self::download_models_sync(&model_dir)?;

        let version_file = model_dir.join("version.txt");
        std::fs::write(&version_file, Self::MODEL_VERSION)
            .map_err(|e| EmotionPredictorError::Io(e.to_string()))?;

        Ok(format!("Models downloaded successfully (version {})", Self::MODEL_VERSION))
    }

    fn get_cache_directory() -> Result<std::path::PathBuf, EmotionPredictorError> {
        let base_path = std::env::current_exe()
            .or_else(|_| std::env::current_dir())
            .unwrap_or_else(|_| ".".into());

        let cache_dir = base_path
            .parent()
            .unwrap_or(Path::new("."))
            .join("npc_models_cache");

        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| EmotionPredictorError::Io(e.to_string()))?;

        Ok(cache_dir)
    }

    fn models_exist_and_valid(model_dir: &Path) -> bool {
        let version_file = model_dir.join("version.txt");

        if let Ok(cached_version) = std::fs::read_to_string(&version_file) {
            if cached_version.trim() != Self::MODEL_VERSION {
                return false;
            }
        } else {
            return false;
        }

        model_dir.join("model.onnx").exists() &&
        model_dir.join("tokenizer.json").exists() &&
        Self::is_placeholder_file(&model_dir.join("model.onnx"))
            .map(|is_placeholder| !is_placeholder)
            .unwrap_or(false)
    }

    fn cleanup_old_versions(cache_dir: &Path) -> Result<(), EmotionPredictorError> {
        if let Ok(entries) = std::fs::read_dir(cache_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with("NPC-Prediction-Model-") && name != format!("NPC-Prediction-Model-{}", Self::MODEL_VERSION) {
                        eprintln!("Cleaning up old model version: {}", name);
                        if let Err(e) = std::fs::remove_dir_all(entry.path()) {
                            eprintln!("Warning: Failed to remove old models: {}", e);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn download_models_sync(model_dir: &Path) -> Result<(), EmotionPredictorError> {
        eprintln!("NPC Neural Affect Matrix: Downloading models for first-time use...");

        std::fs::create_dir_all(model_dir)
            .map_err(|e| EmotionPredictorError::Io(e.to_string()))?;

        let base_url = "https://huggingface.co/Mavdol/NPC-Valence-Arousal-Prediction-ONNX/resolve/main";
        let files = ["model.onnx", "tokenizer.json", "config.json", "vocab.txt"];

        for file in &files {
            let url = format!("{}/{}", base_url, file);
            let file_path = model_dir.join(file);

            if file_path.exists() {
                continue;
            }

            eprintln!("Downloading {}...", file);

            let file_path_str = file_path.to_str()
                .ok_or_else(|| EmotionPredictorError::InvalidInput(
                    format!("Invalid file path for {}: contains non-UTF8 characters", file)
                ))?;

            let response = std::process::Command::new("curl")
                .args(&["-L", "-o", file_path_str, &url])
                .output()
                .map_err(|e| EmotionPredictorError::ModelLoading(format!("Failed to download {}: {}", file, e)))?;

            if !response.status.success() {
                return Err(EmotionPredictorError::ModelLoading(
                    format!("Failed to download {}: {}", file, String::from_utf8_lossy(&response.stderr))
                ));
            }
        }

        eprintln!("NPC Neural Affect Matrix: Models downloaded successfully!");
        Ok(())
    }


    pub fn predict_emotion(&mut self, text: &str) -> Result<EmotionPrediction, EmotionPredictorError> {

        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| EmotionPredictorError::Tokenizer(format!("Tokenization error: {}", e)))?;

        let mut token_ids = encoding.get_ids().to_vec();
        let mut attention_mask = encoding.get_attention_mask().to_vec();

        if token_ids.len() > self.max_length {
            token_ids.truncate(self.max_length);
            attention_mask.truncate(self.max_length);
        } else {
            while token_ids.len() < self.max_length {
                token_ids.push(0);
                attention_mask.push(0);
            }
        }

        let input_ids: Vec<i64> = token_ids.iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();

        let input_ids_value = Value::from_array(([1, self.max_length], input_ids))?;
        let attention_mask_value = Value::from_array(([1, self.max_length], attention_mask))?;

        let outputs = self.session.run(inputs![
            "input_ids" => input_ids_value,
            "attention_mask" => attention_mask_value
        ])
            .map_err(|e| EmotionPredictorError::Inference(format!("Model inference failed: {}", e)))?;

        let output = &outputs[0];

        let (shape, data) = output.try_extract_tensor::<f32>()
            .map_err(|e| EmotionPredictorError::Inference(format!("Failed to extract output: {}", e)))?;

        let predictions = Array2::from_shape_vec((shape[0] as usize, shape[1] as usize), data.to_vec())
            .map_err(|e| EmotionPredictorError::ArrayShape(format!("Failed to create predictions array: {}", e)))?;

        if predictions.shape() != &[1, 2] {
            return Err(EmotionPredictorError::Inference(
                format!("Unexpected output shape: {:?}, expected [1, 2]", predictions.shape())
            ));
        }

        let valence = (predictions[[0, 0]] * 100.0).round() / 100.0;
        let arousal = (predictions[[0, 1]] * 100.0).round() / 100.0;

        Ok(EmotionPrediction::new(valence, arousal))
    }



    pub fn load_tokenizer_with_fallback(tokenizer_path: &Path) -> Result<Tokenizer, EmotionPredictorError> {
        if tokenizer_path.exists() && !Self::is_placeholder_file(tokenizer_path)? {
            match Tokenizer::from_file(tokenizer_path) {
                Ok(tokenizer) => return Ok(tokenizer),
                Err(e) => {
                    eprintln!("Warning: Failed to load tokenizer from {}: {}", tokenizer_path.display(), e);
                    eprintln!("Falling back to simple tokenizer...");
                }
            }
        }

        Self::create_fallback_tokenizer()
    }

    pub fn create_fallback_tokenizer() -> Result<Tokenizer, EmotionPredictorError> {
        return Err(EmotionPredictorError::Tokenizer(
            "Tokenizer file is a placeholder. Please provide a real tokenizer.json file or download the actual model using 'cargo run --bin download-models'.".to_string()
        ));
    }

    pub fn is_placeholder_file(file_path: &Path) -> Result<bool, EmotionPredictorError> {
        if !file_path.exists() {
            return Ok(false);
        }

        if file_path.extension().and_then(|s| s.to_str()) == Some("onnx") {
            let metadata = std::fs::metadata(file_path)?;
            return Ok(metadata.len() < 1000);
        }

        let content = std::fs::read_to_string(file_path)?;
        Ok(content.trim() == "placeholder model data")
    }
}


