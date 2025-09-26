use npc_neural_affect_matrix::utils::SetupUtils;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    SetupUtils::ensure_models().await?;
    Ok(())
}
