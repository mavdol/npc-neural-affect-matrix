pub mod config;
pub mod modules;
pub mod api;

pub use config::{NpcConfig, Identity, PersonalityTraits, MemoryConfig};
pub use modules::memory::{MemoryStore, MemoryRecord, MemoryEmotionEvaluator};
pub use modules::emotion::{EmotionPredictor, EmotionPrediction, EmotionPredictorError};
