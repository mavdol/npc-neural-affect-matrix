use ndarray::Array2;
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;
use tokenizers::Tokenizer;

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
        ort::init().with_name("emotion_prediction").commit()?;

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
        std::fs::write(&version_file, Self::MODEL_VERSION).map_err(|e| EmotionPredictorError::Io(e.to_string()))?;

        Ok(format!(
            "Models downloaded successfully (version {})",
            Self::MODEL_VERSION
        ))
    }

    fn get_cache_directory() -> Result<std::path::PathBuf, EmotionPredictorError> {
        let base_path = std::env::current_exe()
            .or_else(|_| std::env::current_dir())
            .unwrap_or_else(|_| ".".into());

        let cache_dir = base_path.parent().unwrap_or(Path::new(".")).join("npc_models_cache");

        std::fs::create_dir_all(&cache_dir).map_err(|e| EmotionPredictorError::Io(e.to_string()))?;

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

        model_dir.join("model.onnx").exists()
            && model_dir.join("tokenizer.json").exists()
            && Self::is_placeholder_file(&model_dir.join("model.onnx"))
                .map(|is_placeholder| !is_placeholder)
                .unwrap_or(false)
    }

    fn cleanup_old_versions(cache_dir: &Path) -> Result<(), EmotionPredictorError> {
        if let Ok(entries) = std::fs::read_dir(cache_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with("NPC-Prediction-Model-")
                        && name != format!("NPC-Prediction-Model-{}", Self::MODEL_VERSION)
                    {
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

        std::fs::create_dir_all(model_dir).map_err(|e| EmotionPredictorError::Io(e.to_string()))?;

        let base_url = "https://huggingface.co/Mavdol/NPC-Valence-Arousal-Prediction-ONNX/resolve/main";
        let files = ["model.onnx", "tokenizer.json", "config.json", "vocab.txt"];

        for file in &files {
            let url = format!("{}/{}", base_url, file);
            let file_path = model_dir.join(file);

            if file_path.exists() {
                continue;
            }

            eprintln!("Downloading {}...", file);

            let file_path_str = file_path.to_str().ok_or_else(|| {
                EmotionPredictorError::InvalidInput(format!(
                    "Invalid file path for {}: contains non-UTF8 characters",
                    file
                ))
            })?;

            let response = std::process::Command::new("curl")
                .args(&["-L", "-o", file_path_str, &url])
                .output()
                .map_err(|e| EmotionPredictorError::ModelLoading(format!("Failed to download {}: {}", file, e)))?;

            if !response.status.success() {
                return Err(EmotionPredictorError::ModelLoading(format!(
                    "Failed to download {}: {}",
                    file,
                    String::from_utf8_lossy(&response.stderr)
                )));
            }
        }

        eprintln!("NPC Neural Affect Matrix: Models downloaded successfully!");
        Ok(())
    }

    pub fn predict_emotion(&mut self, text: &str) -> Result<EmotionPrediction, EmotionPredictorError> {
        let encoding = self
            .tokenizer
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

        let outputs = self
            .session
            .run(inputs![
                "input_ids" => input_ids_value,
                "attention_mask" => attention_mask_value
            ])
            .map_err(|e| EmotionPredictorError::Inference(format!("Model inference failed: {}", e)))?;

        let output = &outputs[0];

        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| EmotionPredictorError::Inference(format!("Failed to extract output: {}", e)))?;

        let predictions = Array2::from_shape_vec((shape[0] as usize, shape[1] as usize), data.to_vec())
            .map_err(|e| EmotionPredictorError::ArrayShape(format!("Failed to create predictions array: {}", e)))?;

        if predictions.shape() != &[1, 2] {
            return Err(EmotionPredictorError::Inference(format!(
                "Unexpected output shape: {:?}, expected [1, 2]",
                predictions.shape()
            )));
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
                    eprintln!(
                        "Warning: Failed to load tokenizer from {}: {}",
                        tokenizer_path.display(),
                        e
                    );
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

#[cfg(test)]
mod tests {
    use super::{EmotionPrediction, EmotionPredictor, EmotionPredictorError};
    use crate::_test_mock::emotion_mock::{
        create_mock_model_directory, EmotionPredict, MockEmotionPredictor, TestEmotionData,
    };
    use std::path::Path;

    #[test]
    fn test_io_error_conversion() {
        let io_error = std::fs::read_to_string("/nonexistent/path/file.txt").unwrap_err();
        let emotion_error: EmotionPredictorError = io_error.into();
        assert!(matches!(emotion_error, EmotionPredictorError::Io(_)));
    }

    #[test]
    fn test_tokenizer_error() {
        let mut mock_predictor = MockEmotionPredictor::new().with_response(
            "invalid_tokenizer_input",
            Err(EmotionPredictorError::Tokenizer("Invalid tokenizer input".to_string())),
        );

        let result = mock_predictor.predict_emotion("invalid_tokenizer_input");
        match result {
            Err(EmotionPredictorError::Tokenizer(_)) => {}
            Err(e) => panic!("Expected Tokenizer error, got: {:?}", e),
            Ok(_) => panic!("Expected error, but got Ok"),
        }
    }

    #[test]
    fn test_model_loading_error() {
        let mut mock_predictor = MockEmotionPredictor::new().with_response(
            "model_loading_test",
            Err(EmotionPredictorError::ModelLoading("Failed to load model".to_string())),
        );

        let result = mock_predictor.predict_emotion("model_loading_test");
        match result {
            Err(EmotionPredictorError::ModelLoading(_)) => {}
            Err(e) => panic!("Expected ModelLoading error, got: {:?}", e),
            Ok(_) => panic!("Expected error, but got Ok"),
        }
    }

    #[test]
    fn test_onnx_runtime_error_conversion() {
        let ort_error = ort::Error::new("ONNX Runtime error");
        let emotion_error: EmotionPredictorError = ort_error.into();
        assert!(matches!(emotion_error, EmotionPredictorError::OnnxRuntime(_)));
    }

    #[test]
    fn test_error_display_formatting() {
        let inference_error = EmotionPredictorError::Inference("Inference failed".to_string());
        assert!(format!("{}", inference_error).contains("Inference failed"));

        let input_error = EmotionPredictorError::InvalidInput("Invalid input".to_string());
        assert!(format!("{}", input_error).contains("Invalid input"));

        let array_error = EmotionPredictorError::ArrayShape("Shape mismatch".to_string());
        assert!(format!("{}", array_error).contains("Shape mismatch"));
    }

    #[test]
    fn test_emotion_prediction_new() {
        let prediction = EmotionPrediction::new(0.52, -0.32);
        assert_eq!(prediction.valence, 0.52);
        assert_eq!(prediction.arousal, -0.32);
    }

    #[test]
    fn test_emotion_prediction_values() {
        let prediction = EmotionPrediction::new(0.72, -0.20);
        let (valence, arousal) = prediction.values();
        assert_eq!(valence, 0.72);
        assert_eq!(arousal, -0.20);
    }

    #[test]
    fn test_emotion_prediction_equality() {
        let pred1 = EmotionPrediction::new(0.5, -0.3);
        let pred2 = EmotionPrediction::new(0.5, -0.3);
        let pred3 = EmotionPrediction::new(0.5, 0.3);

        assert_eq!(pred1.valence, pred2.valence);
        assert_eq!(pred1.arousal, pred2.arousal);
        assert_ne!(pred1.arousal, pred3.arousal);
    }

    #[test]
    fn test_mock_emotion_predictor_positive_emotions() {
        let mut predictor = MockEmotionPredictor::positive();

        for text in TestEmotionData::happy_texts() {
            let result = predictor.predict_emotion(text);
            assert!(result.is_ok());
            let prediction = result.unwrap();
            assert!(prediction.valence > 0.5, "Expected positive valence for: {}", text);
            assert!(prediction.arousal > 0.0, "Expected positive arousal for: {}", text);
        }
    }

    #[test]
    fn test_mock_emotion_predictor_negative_emotions() {
        let mut predictor = MockEmotionPredictor::negative();

        for text in TestEmotionData::sad_texts() {
            let result = predictor.predict_emotion(text);
            assert!(result.is_ok());
            let prediction = result.unwrap();
            assert!(prediction.valence < 0.0, "Expected negative valence for: {}", text);
        }
    }

    #[test]
    fn test_mock_emotion_predictor_custom_responses() {
        let mut predictor = MockEmotionPredictor::new()
            .with_response("I'm happy!", Ok(EmotionPrediction::new(0.9, 0.7)))
            .with_response("I'm sad", Ok(EmotionPrediction::new(-0.8, -0.3)))
            .with_response(
                "Error text",
                Err(EmotionPredictorError::Inference("Test error".to_string())),
            );

        let happy_result = predictor.predict_emotion("I'm happy!");
        assert!(happy_result.is_ok());
        let happy_pred = happy_result.unwrap();
        assert_eq!(happy_pred.valence, 0.9);
        assert_eq!(happy_pred.arousal, 0.7);

        let sad_result = predictor.predict_emotion("I'm sad");
        assert!(sad_result.is_ok());
        let sad_pred = sad_result.unwrap();
        assert_eq!(sad_pred.valence, -0.8);
        assert_eq!(sad_pred.arousal, -0.3);

        let error_result = predictor.predict_emotion("Error text");
        assert!(error_result.is_err());
        if let Err(EmotionPredictorError::Inference(msg)) = error_result {
            assert_eq!(msg, "Test error");
        } else {
            panic!("Expected Inference error");
        }

        let default_result = predictor.predict_emotion("unknown text");
        assert!(default_result.is_ok());
        let default_pred = default_result.unwrap();
        assert_eq!(default_pred.valence, 0.0);
        assert_eq!(default_pred.arousal, 0.0);
    }

    #[test]
    fn test_mock_predictor_performance() {
        let mut predictor = MockEmotionPredictor::neutral();
        let start = std::time::Instant::now();

        for i in 0..1000 {
            let text = format!("Test text number {}", i);
            let result = predictor.predict_emotion(&text);
            assert!(result.is_ok());
        }

        let duration = start.elapsed();
        assert!(duration.as_millis() < 100, "Mock predictor too slow: {:?}", duration);
    }

    #[test]
    fn test_emotion_prediction_edge_cases() {
        let mut predictor = MockEmotionPredictor::new();

        let result = predictor.predict_emotion("");
        assert!(result.is_ok());

        let long_text = "a".repeat(10000);
        let result = predictor.predict_emotion(&long_text);
        assert!(result.is_ok());

        let special_text = "!@#$%^&*()_+{}|:<>?[];',./~`";
        let result = predictor.predict_emotion(special_text);
        assert!(result.is_ok());

        let unicode_text = "Hello 世界 🌍 café naïve résumé";
        let result = predictor.predict_emotion(unicode_text);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_and_download_models() {
        let result = EmotionPredictor::check_and_download_models();

        match result {
            Ok(message) => {
                assert!(message.contains("Models") || message.contains("downloaded") || message.contains("up to date"));
            }
            Err(e) => {
                assert!(matches!(
                    e,
                    EmotionPredictorError::ModelLoading(_) | EmotionPredictorError::Io(_)
                ));
            }
        }
    }

    #[test]
    fn test_load_tokenizer_with_fallback_existing_file() {
        if let Ok(temp_dir) = create_mock_model_directory() {
            let tokenizer_path = temp_dir.join("tokenizer.json");

            let result = EmotionPredictor::load_tokenizer_with_fallback(&tokenizer_path);

            let _ = std::fs::remove_dir_all(&temp_dir);

            assert!(
                result.is_ok(),
                "Expected tokenizer to load successfully, but got error: {:?}",
                result
            );
        }
    }

    #[test]
    fn test_load_tokenizer_with_fallback_invalid_file() {
        let temp_dir = std::env::temp_dir().join(format!("test_invalid_tokenizer_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).unwrap();

        let tokenizer_path = temp_dir.join("tokenizer.json");

        std::fs::write(&tokenizer_path, "invalid json content").unwrap();

        let result = EmotionPredictor::load_tokenizer_with_fallback(&tokenizer_path);

        let _ = std::fs::remove_dir_all(&temp_dir);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, EmotionPredictorError::Tokenizer(_)));
        }
    }

    #[test]
    fn test_load_tokenizer_with_fallback_nonexistent_file() {
        let nonexistent_path = Path::new("/nonexistent/tokenizer.json");
        let result = EmotionPredictor::load_tokenizer_with_fallback(nonexistent_path);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, EmotionPredictorError::Tokenizer(_)));
        }
    }

    #[test]
    fn test_create_fallback_tokenizer() {
        let result = EmotionPredictor::create_fallback_tokenizer();

        assert!(result.is_err());
        if let Err(EmotionPredictorError::Tokenizer(msg)) = result {
            assert!(msg.contains("placeholder"));
            assert!(msg.contains("download-models"));
        } else {
            panic!("Expected Tokenizer error");
        }
    }

    #[test]
    fn test_is_placeholder_file_nonexistent() {
        let nonexistent_path = Path::new("/nonexistent/file.txt");
        let result = EmotionPredictor::is_placeholder_file(nonexistent_path);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), false);
    }
}
