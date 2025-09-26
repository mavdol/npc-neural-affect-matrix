use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use sha2::{Sha256, Digest};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DownloadError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Progress bar template error: {0}")]
    ProgressTemplate(#[from] indicatif::style::TemplateError),

    #[error("File verification failed: expected {expected}, got {actual}")]
    VerificationFailed { expected: String, actual: String },

    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}

pub type DownloadResult<T> = Result<T, DownloadError>;

pub struct ModelDownloader {
    client: Client,
    chunk_size: usize,
}

impl Default for ModelDownloader {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelDownloader {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            chunk_size: 8192,
        }
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    pub async fn download_file<P: AsRef<Path>>(
        &self,
        url: &str,
        destination: P,
        expected_sha256: Option<&str>,
    ) -> DownloadResult<()> {
        let destination = destination.as_ref();

        if let Some(parent) = destination.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let response = self.client.head(url).send().await?;
        let total_size = response
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|ct_len| ct_len.to_str().ok())
            .and_then(|ct_len| ct_len.parse::<u64>().ok())
            .unwrap_or(0);

        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
                .progress_chars("#>-"),
        );
        pb.set_message(format!("Downloading {}", destination.file_name().unwrap_or_default().to_string_lossy()));

        let mut response = self.client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(DownloadError::InvalidResponse(format!(
                "HTTP {}: {}",
                response.status(),
                response.text().await.unwrap_or_else(|_| "Unknown error".to_string())
            )));
        }

        let mut file = File::create(destination).await?;
        let mut hasher = expected_sha256.map(|_| Sha256::new());
        let mut downloaded = 0u64;

        while let Some(chunk) = response.chunk().await? {
            file.write_all(&chunk).await?;

            if let Some(ref mut hasher) = hasher {
                hasher.update(&chunk);
            }

            downloaded += chunk.len() as u64;
            pb.set_position(downloaded);
        }

        file.flush().await?;
        pb.finish_with_message(format!("Downloaded {}", destination.file_name().unwrap_or_default().to_string_lossy()));

        if let Some(expected_sha256) = expected_sha256 {
            if let Some(hasher) = hasher {
                let actual_hash = hex::encode(hasher.finalize());
                if actual_hash != expected_sha256 {
                    let _ = tokio::fs::remove_file(destination).await;
                    return Err(DownloadError::VerificationFailed {
                        expected: expected_sha256.to_string(),
                        actual: actual_hash,
                    });
                }
            }
        }

        Ok(())
    }

    pub async fn download_files<P: AsRef<Path>>(
        &self,
        downloads: Vec<(&str, P, Option<&str>)>,
    ) -> DownloadResult<()> {
        let tasks = downloads.into_iter().map(|(url, dest, sha256)| {
            let downloader = &self;
            async move { downloader.download_file(url, dest, sha256).await }
        });

        let results = futures_util::future::join_all(tasks).await;

        for result in results {
            result?;
        }

        Ok(())
    }
}
