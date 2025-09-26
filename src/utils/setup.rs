use std::path::Path;
use serde::Deserialize;
use super::model_manager::{ModelManager, create_valence_arousal_model_info};

#[derive(Debug, Deserialize)]
pub struct SimpleConfig {
    pub identity: SimpleIdentity,
}

#[derive(Debug, Deserialize)]
pub struct SimpleIdentity {
    pub name: String,
}

pub struct SetupUtils;

impl SetupUtils {
    pub async fn ensure_models() -> Result<(), Box<dyn std::error::Error>> {
        let model_manager = ModelManager::with_default_dir();
        let model_info = create_valence_arousal_model_info();

        println!("Downloading models...");
        model_manager.ensure_model(&model_info).await?;
        println!("Models ready!");

        Ok(())
    }

    pub async fn run_setup() -> Result<(), Box<dyn std::error::Error>> {
        if !Path::new("models").exists() {
            Self::ensure_models().await?;
        } else {
            println!("✅ Models directory found.");
        }

        Ok(())
    }
}

pub struct BuildSetup;

impl BuildSetup {
    pub fn run_download_models() -> Result<(), Box<dyn std::error::Error>> {
        println!("cargo:warning=Step 1: Downloading models...");

        let model_manager = ModelManager::with_default_dir();
        let model_info = create_valence_arousal_model_info();

        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            model_manager.ensure_model(&model_info).await
        })?;

        println!("cargo:warning=✅ Models downloaded successfully!");
        Ok(())
    }

    pub fn run_build_setup() -> Result<(), Box<dyn std::error::Error>> {
        println!("cargo:warning=🚀 Starting NPC Neural Affect Matrix setup...");

        if std::env::var("SKIP_SETUP").is_ok() {
            println!("cargo:warning=⏭️  Skipping setup (SKIP_SETUP is set)");
            return Ok(());
        }

        if !Path::new("models").exists() {
            if let Err(e) = Self::run_download_models() {
                println!("cargo:warning=❌ Failed to download models: {}", e);
                println!("cargo:warning=   Run manually: cargo run --bin download_models");
            }
        } else {
            println!("cargo:warning=✅ Models directory found.");
        }


        println!("cargo:warning=🎉 Setup completed!");
        println!("cargo:rerun-if-changed=models");
        println!("cargo:rerun-if-changed=npc_config.toml");

        Ok(())
    }
}
