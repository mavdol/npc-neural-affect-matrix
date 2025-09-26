use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use crate::utils::downloader::{ModelDownloader, DownloadError};

#[derive(Error, Debug)]
pub enum ModelManagerError {
    #[error("Download error: {0}")]
    Download(#[from] DownloadError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Model not found at path: {0}")]
    ModelNotFound(String),

    #[error("Invalid model configuration: {0}")]
    InvalidConfig(String),
}

pub type ModelManagerResult<T> = Result<T, ModelManagerError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub huggingface_repo: String,
    pub files: Vec<ModelFile>,
    pub config: ModelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub filename: String,
    pub url: String,
    pub sha256: Option<String>,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub task: String,
    pub languages: Vec<String>,
    pub input_dim: Option<usize>,
    pub output_dim: Option<usize>,
    pub max_sequence_length: usize,
}

pub struct ModelManager {
    models_dir: PathBuf,
    downloader: ModelDownloader,
}

impl ModelManager {
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Self {
        Self {
            models_dir: models_dir.as_ref().to_path_buf(),
            downloader: ModelDownloader::new(),
        }
    }

    pub fn default_models_dir() -> PathBuf {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("models")
    }

    pub fn with_default_dir() -> Self {
        Self::new(Self::default_models_dir())
    }

    pub fn model_path(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(model_name)
    }

    pub async fn model_exists(&self, model_info: &ModelInfo) -> bool {
        let model_dir = self.model_path(&model_info.name);

        if !model_dir.exists() {
            return false;
        }

        for file in &model_info.files {
            let file_path = model_dir.join(&file.filename);
            if !file_path.exists() {
                return false;
            }
        }

        let info_path = model_dir.join("model_info.json");
        info_path.exists()
    }

    pub async fn ensure_model(&self, model_info: &ModelInfo) -> ModelManagerResult<PathBuf> {
        let model_dir = self.model_path(&model_info.name);

        if self.model_exists(model_info).await {
            return Ok(model_dir);
        }

        println!("Downloading model '{}'...", model_info.name);
        self.download_model(model_info).await?;

        Ok(model_dir)
    }


    pub async fn download_model(&self, model_info: &ModelInfo) -> ModelManagerResult<()> {
        let model_dir = self.model_path(&model_info.name);

        tokio::fs::create_dir_all(&model_dir).await?;

        let downloads: Vec<_> = model_info
            .files
            .iter()
            .map(|file| {
                let dest_path = model_dir.join(&file.filename);
                (file.url.as_str(), dest_path, file.sha256.as_deref())
            })
            .collect();


        self.downloader.download_files(downloads).await?;


        let info_path = model_dir.join("model_info.json");
        let info_json = serde_json::to_string_pretty(model_info)?;
        tokio::fs::write(info_path, info_json).await?;

        println!("Model '{}' downloaded successfully to {:?}", model_info.name, model_dir);
        Ok(())
    }

    pub async fn load_model_info<P: AsRef<Path>>(&self, model_path: P) -> ModelManagerResult<ModelInfo> {
        let info_path = model_path.as_ref().join("model_info.json");

        if !info_path.exists() {
            return Err(ModelManagerError::ModelNotFound(info_path.display().to_string()));
        }

        let info_json = tokio::fs::read_to_string(info_path).await?;
        let model_info: ModelInfo = serde_json::from_str(&info_json)?;

        Ok(model_info)
    }

    pub async fn list_models(&self) -> ModelManagerResult<Vec<ModelInfo>> {
        let mut models = Vec::new();

        if !self.models_dir.exists() {
            return Ok(models);
        }

        let mut entries = tokio::fs::read_dir(&self.models_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_dir() {
                if let Ok(model_info) = self.load_model_info(entry.path()).await {
                    models.push(model_info);
                }
            }
        }

        Ok(models)
    }

    pub async fn remove_model(&self, model_name: &str) -> ModelManagerResult<()> {
        let model_dir = self.model_path(model_name);

        if model_dir.exists() {
            tokio::fs::remove_dir_all(model_dir).await?;
            println!("Model '{}' removed successfully", model_name);
        }

        Ok(())
    }
}

pub fn create_valence_arousal_model_info() -> ModelInfo {
    ModelInfo {
        name: "NPC-Valence-Arousal-Prediction-ONNX".to_string(),
        version: "1.0.0".to_string(),
        huggingface_repo: "Mavdol/NPC-Valence-Arousal-Prediction-ONNX".to_string(),
        files: vec![
            ModelFile {
                filename: "config.json".to_string(),
                url: "https://huggingface.co/Mavdol/NPC-Valence-Arousal-Prediction-ONNX/resolve/main/config.json".to_string(),
                sha256: None,
                size_bytes: None,
            },
            ModelFile {
                filename: "model.onnx".to_string(),
                url: "https://huggingface.co/Mavdol/NPC-Valence-Arousal-Prediction-ONNX/resolve/main/model.onnx".to_string(),
                sha256: None,
                size_bytes: None,
            },
            ModelFile {
                filename: "tokenizer.json".to_string(),
                url: "https://huggingface.co/Mavdol/NPC-Valence-Arousal-Prediction-ONNX/resolve/main/tokenizer.json".to_string(),
                sha256: None,
                size_bytes: None,
            },
            ModelFile {
                filename: "tokenizer_config.json".to_string(),
                url: "https://huggingface.co/Mavdol/NPC-Valence-Arousal-Prediction-ONNX/resolve/main/tokenizer_config.json".to_string(),
                sha256: None,
                size_bytes: None,
            },
            ModelFile {
                filename: "vocab.txt".to_string(),
                url: "https://huggingface.co/Mavdol/NPC-Valence-Arousal-Prediction-ONNX/resolve/main/special_tokens_map.json".to_string(),
                sha256: None,
                size_bytes: None,
            },
            ModelFile {
                filename: "special_tokens_map.json".to_string(),
                url: "https://huggingface.co/Mavdol/NPC-Valence-Arousal-Prediction-ONNX/resolve/main/special_tokens_map.json".to_string(),
                sha256: None,
                size_bytes: None,
            },
        ],
        config: ModelConfig {
            model_type: "distilbert".to_string(),
            task: "valence-arousal-prediction".to_string(),
            languages: vec![
                "english".to_string(),
            ],
            input_dim: Some(768),
            output_dim: Some(2),
            max_sequence_length: 512,
        },
    }
}

