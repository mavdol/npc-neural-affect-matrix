use std::path::Path;
use std::env;
use reqwest::Client;
use sha2::{Sha256, Digest};
use std::fs::File;
use std::io::Write;

const MODEL_REPO: &str = "Mavdol/NPC-Valence-Arousal-Prediction-ONNX";
const MODEL_NAME: &str = "NPC-Valence-Arousal-Prediction-ONNX";
const MODEL_VERSION: &str = "1.0.0";
const MODELS_DIR: &str = "models";
const CONFIG_FILE: &str = "npc_config.toml";

#[derive(Debug, Clone)]
struct ModelFile {
    filename: String,
    url: String,
    sha256: Option<String>,
}

#[derive(Debug, Clone)]
struct ModelInfo {
    name: String,
    version: String,
    huggingface_repo: String,
    files: Vec<ModelFile>,
}

fn create_model_info() -> ModelInfo {
    let base_url = format!("https://huggingface.co/{}/resolve/main", MODEL_REPO);
    let files = vec![
        "config.json",
        "model.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json"
    ];

    ModelInfo {
        name: MODEL_NAME.to_string(),
        version: MODEL_VERSION.to_string(),
        huggingface_repo: MODEL_REPO.to_string(),
        files: files.into_iter()
            .map(|filename| ModelFile {
                filename: filename.to_string(),
                url: format!("{}/{}", base_url, filename),
                sha256: None,
            })
            .collect(),
    }
}

async fn download_file(url: &str, destination: &Path, expected_sha256: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:warning=Downloading: {}", destination.file_name().unwrap_or_default().to_string_lossy());

    let client = Client::new();
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(format!("HTTP {}: {}", response.status(), response.text().await.unwrap_or_else(|_| "Unknown error".to_string())).into());
    }

    let mut file = File::create(destination)?;
    let mut hasher = expected_sha256.map(|_| Sha256::new());
    let mut stream = response.bytes_stream();

    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;

        if let Some(ref mut hasher) = hasher {
            hasher.update(&chunk);
        }
    }

    file.flush()?;

    if let Some(expected_sha256) = expected_sha256 {
        if let Some(hasher) = hasher {
            let actual_hash = hex::encode(hasher.finalize());
            if actual_hash != expected_sha256 {
                std::fs::remove_file(destination)?;
                return Err(format!("File verification failed: expected {}, got {}", expected_sha256, actual_hash).into());
            }
        }
    }

    println!("cargo:warning=Downloaded: {}", destination.file_name().unwrap_or_default().to_string_lossy());
    Ok(())
}

fn model_files_exist(model_info: &ModelInfo) -> bool {
    let model_path = Path::new(MODELS_DIR).join(&model_info.name);
    model_path.exists() && model_info.files.iter().all(|file| {
        model_path.join(&file.filename).exists()
    })
}

async fn setup_models() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:warning=Step 1: Downloading models from Hugging Face...");

    let model_info = create_model_info();

    if model_files_exist(&model_info) {
        println!("cargo:warning=✅ Model '{}' already exists, skipping download", model_info.name);
        return Ok(());
    }

    std::fs::create_dir_all(MODELS_DIR)?;
    let model_path = Path::new(MODELS_DIR).join(&model_info.name);
    std::fs::create_dir_all(&model_path)?;

    for file in &model_info.files {
        let file_path = model_path.join(&file.filename);
        if !file_path.exists() {
            download_file(&file.url, &file_path, file.sha256.as_deref()).await?;
        }
    }

    let model_info_json = serde_json::json!({
        "name": model_info.name,
        "version": model_info.version,
        "huggingface_repo": model_info.huggingface_repo,
        "files": model_info.files.iter().map(|f| serde_json::json!({
            "filename": f.filename,
            "url": f.url,
            "sha256": f.sha256
        })).collect::<Vec<_>>(),
        "config": {
            "model_type": "distilbert",
            "task": "valence-arousal-prediction",
            "languages": ["english"],
            "input_dim": 768,
            "output_dim": 2,
            "max_sequence_length": 512
        }
    });

    std::fs::write(model_path.join("model_info.json"), serde_json::to_string_pretty(&model_info_json)?)?;
    println!("cargo:warning=✅ Model '{}' downloaded successfully!", model_info.name);
    Ok(())
}

fn main() {
    println!("cargo:warning=🚀 Starting NPC Neural Affect Matrix setup...");

    if env::var("SKIP_SETUP").is_ok() {
        println!("cargo:warning=⏭️  Skipping setup (SKIP_SETUP is set)");
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();

    if let Err(e) = rt.block_on(setup_models()) {
        println!("cargo:warning=❌ Failed to download models: {}", e);
        println!("cargo:warning=   Run manually: cargo run --bin download-models");
    }

    println!("cargo:warning=🎉 Setup completed!");
    println!("cargo:rerun-if-changed={}", MODELS_DIR);
    println!("cargo:rerun-if-changed={}", CONFIG_FILE);
}
