use crate::config::NpcConfig;
use crate::modules::emotion::{EmotionPrediction, EmotionPredictorError};
use crate::modules::memory::store::{MemoryRecord, MemoryStore};
use std::collections::HashMap;

pub struct MockMemoryEmotionEvaluator {
    pub config: NpcConfig,
    pub source_id: Option<String>,
    pub npc_id: String,
    predict_responses: HashMap<String, Result<EmotionPrediction, EmotionPredictorError>>,
    default_predict_response: Result<EmotionPrediction, EmotionPredictorError>,
    current_emotion_response: Result<EmotionPrediction, EmotionPredictorError>,
    source_emotion_responses: HashMap<String, Result<EmotionPrediction, EmotionPredictorError>>,
}

impl MockMemoryEmotionEvaluator {
    pub fn new(config: NpcConfig, source_id: Option<String>) -> Result<Self, EmotionPredictorError> {
        let npc_id = "test-npc-id".to_string();
        Ok(Self {
            config,
            source_id,
            npc_id,
            predict_responses: HashMap::new(),
            default_predict_response: Ok(EmotionPrediction::new(0.0, 0.0)),
            current_emotion_response: Ok(EmotionPrediction::new(0.0, 0.0)),
            source_emotion_responses: HashMap::new(),
        })
    }

    pub fn new_with_id(
        config: NpcConfig,
        source_id: Option<String>,
        npc_id: String,
    ) -> Result<Self, EmotionPredictorError> {
        Ok(Self {
            config,
            source_id,
            npc_id,
            predict_responses: HashMap::new(),
            default_predict_response: Ok(EmotionPrediction::new(0.0, 0.0)),
            current_emotion_response: Ok(EmotionPrediction::new(0.0, 0.0)),
            source_emotion_responses: HashMap::new(),
        })
    }

    pub fn with_predict_response(
        mut self,
        text: &str,
        response: Result<EmotionPrediction, EmotionPredictorError>,
    ) -> Self {
        self.predict_responses.insert(text.to_string(), response);
        self
    }

    pub fn with_default_predict_response(mut self, response: Result<EmotionPrediction, EmotionPredictorError>) -> Self {
        self.default_predict_response = response;
        self
    }

    pub fn with_current_emotion_response(mut self, response: Result<EmotionPrediction, EmotionPredictorError>) -> Self {
        self.current_emotion_response = response;
        self
    }

    pub fn with_source_emotion_response(
        mut self,
        source_id: &str,
        response: Result<EmotionPrediction, EmotionPredictorError>,
    ) -> Self {
        self.source_emotion_responses.insert(source_id.to_string(), response);
        self
    }

    pub fn positive() -> Self {
        let config = NpcConfig::default();
        Self::new(config, Some("test-source".to_string()))
            .unwrap()
            .with_default_predict_response(Ok(EmotionPrediction::new(0.8, 0.6)))
            .with_current_emotion_response(Ok(EmotionPrediction::new(0.7, 0.5)))
    }

    pub fn negative() -> Self {
        let config = NpcConfig::default();
        Self::new(config, Some("test-source".to_string()))
            .unwrap()
            .with_default_predict_response(Ok(EmotionPrediction::new(-0.7, -0.4)))
            .with_current_emotion_response(Ok(EmotionPrediction::new(-0.6, -0.3)))
    }

    pub fn neutral() -> Self {
        let config = NpcConfig::default();
        Self::new(config, Some("test-source".to_string()))
            .unwrap()
            .with_default_predict_response(Ok(EmotionPrediction::new(0.0, 0.0)))
            .with_current_emotion_response(Ok(EmotionPrediction::new(0.0, 0.0)))
    }

    pub fn error() -> Self {
        let config = NpcConfig::default();
        Self::new(config, Some("test-source".to_string()))
            .unwrap()
            .with_default_predict_response(Err(EmotionPredictorError::Inference(
                "Mock prediction error".to_string(),
            )))
            .with_current_emotion_response(Err(EmotionPredictorError::Inference(
                "Mock current emotion error".to_string(),
            )))
    }

    pub fn predict(&self, text: &str, past_time: i64) -> Result<EmotionPrediction, EmotionPredictorError> {
        let prediction = self
            .predict_responses
            .get(text)
            .cloned()
            .unwrap_or_else(|| self.default_predict_response.clone())?;

        let record = MemoryRecord {
            id: uuid::Uuid::new_v4().to_string(),
            source_id: self.source_id.clone().unwrap_or_else(|| "unknown".to_string()),
            text: text.to_string(),
            valence: prediction.valence,
            arousal: prediction.arousal,
            past_time,
        };

        MemoryStore::insert(&self.npc_id, record)
            .map_err(|e| EmotionPredictorError::Inference(format!("Failed to store memory: {}", e)))?;

        Ok(prediction)
    }

    pub fn calculate_current_emotion(&self) -> Result<EmotionPrediction, EmotionPredictorError> {
        self.current_emotion_response.clone()
    }

    pub fn calculate_current_emotion_towards_source(
        &self,
        source_id: &str,
    ) -> Result<EmotionPrediction, EmotionPredictorError> {
        self.source_emotion_responses
            .get(source_id)
            .cloned()
            .unwrap_or_else(|| self.current_emotion_response.clone())
    }
}

impl Clone for MockMemoryEmotionEvaluator {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            source_id: self.source_id.clone(),
            npc_id: self.npc_id.clone(),
            predict_responses: self.predict_responses.clone(),
            default_predict_response: self.default_predict_response.clone(),
            current_emotion_response: self.current_emotion_response.clone(),
            source_emotion_responses: self.source_emotion_responses.clone(),
        }
    }
}

pub struct TestMemoryData;

impl TestMemoryData {
    pub fn sample_memory_records() -> Vec<MemoryRecord> {
        vec![
            MemoryRecord {
                id: "memory-1".to_string(),
                source_id: "source-1".to_string(),
                text: "Happy conversation".to_string(),
                valence: 0.8,
                arousal: 0.6,
                past_time: 1000,
            },
            MemoryRecord {
                id: "memory-2".to_string(),
                source_id: "source-1".to_string(),
                text: "Sad conversation".to_string(),
                valence: -0.6,
                arousal: -0.3,
                past_time: 2000,
            },
            MemoryRecord {
                id: "memory-3".to_string(),
                source_id: "source-2".to_string(),
                text: "Angry conversation".to_string(),
                valence: -0.7,
                arousal: 0.8,
                past_time: 3000,
            },
            MemoryRecord {
                id: "memory-4".to_string(),
                source_id: "source-2".to_string(),
                text: "Neutral conversation".to_string(),
                valence: 0.0,
                arousal: 0.0,
                past_time: 4000,
            },
        ]
    }

    pub fn happy_memory_record() -> MemoryRecord {
        MemoryRecord {
            id: "happy-1".to_string(),
            source_id: "friend".to_string(),
            text: "We had such a great time together!".to_string(),
            valence: 0.9,
            arousal: 0.7,
            past_time: 500,
        }
    }

    pub fn sad_memory_record() -> MemoryRecord {
        MemoryRecord {
            id: "sad-1".to_string(),
            source_id: "friend".to_string(),
            text: "I'm really disappointed about what happened".to_string(),
            valence: -0.8,
            arousal: -0.2,
            past_time: 1500,
        }
    }

    pub fn angry_memory_record() -> MemoryRecord {
        MemoryRecord {
            id: "angry-1".to_string(),
            source_id: "enemy".to_string(),
            text: "That was completely unfair and wrong!".to_string(),
            valence: -0.9,
            arousal: 0.9,
            past_time: 800,
        }
    }

    pub fn neutral_memory_record() -> MemoryRecord {
        MemoryRecord {
            id: "neutral-1".to_string(),
            source_id: "stranger".to_string(),
            text: "The weather is nice today".to_string(),
            valence: 0.0,
            arousal: 0.0,
            past_time: 2000,
        }
    }

    pub fn expected_combined_emotion() -> EmotionPrediction {
        EmotionPrediction::new(0.2, 0.1) // Slightly positive, low arousal
    }

    pub fn expected_source_emotion() -> EmotionPrediction {
        EmotionPrediction::new(0.5, 0.3) // Moderately positive
    }
}

pub struct MemoryTestHelpers;

impl MemoryTestHelpers {
    pub fn test_npc_id() -> String {
        "test-npc-12345".to_string()
    }

    pub fn test_npc_ids(count: usize) -> Vec<String> {
        (0..count).map(|i| format!("test-npc-{}", i)).collect()
    }

    pub fn clear_test_npc_memory(npc_id: &str) -> Result<(), String> {
        MemoryStore::clear(&npc_id.to_string())
    }

    pub fn setup_test_memories(npc_id: &str) -> Result<(), String> {
        Self::clear_test_npc_memory(npc_id)?;
        let records = TestMemoryData::sample_memory_records();
        MemoryStore::import(&npc_id.to_string(), records)
    }

    pub fn validate_memory_record(record: &MemoryRecord) -> Result<(), String> {
        if record.id.is_empty() {
            return Err("Memory record ID cannot be empty".to_string());
        }
        if record.valence < -1.0 || record.valence > 1.0 {
            return Err(format!(
                "Invalid valence: {} (must be between -1.0 and 1.0)",
                record.valence
            ));
        }
        if record.arousal < -1.0 || record.arousal > 1.0 {
            return Err(format!(
                "Invalid arousal: {} (must be between -1.0 and 1.0)",
                record.arousal
            ));
        }
        Ok(())
    }

    pub fn create_valid_memory_record(
        id: &str,
        source_id: &str,
        text: &str,
        valence: f32,
        arousal: f32,
        past_time: i64,
    ) -> Result<MemoryRecord, String> {
        let record = MemoryRecord {
            id: id.to_string(),
            source_id: source_id.to_string(),
            text: text.to_string(),
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(-1.0, 1.0),
            past_time,
        };
        Self::validate_memory_record(&record)?;
        Ok(record)
    }
}
