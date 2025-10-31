use crate::{EmotionPrediction, NpcConfig};
use crate::{EmotionPredictor, EmotionPredictorError};
use crate::{MemoryRecord, MemoryStore};
use std::f32::consts::E;
use uuid::Uuid;

#[derive(Clone)]
pub struct MemoryEmotionEvaluator {
    pub config: NpcConfig,
    pub source_id: Option<String>,
    pub npc_id: String,
}

impl MemoryEmotionEvaluator {
    pub fn new(config: NpcConfig, source_id: Option<String>) -> Result<Self, EmotionPredictorError> {
        let npc_id = uuid::Uuid::new_v4().to_string();
        Ok(Self {
            config,
            source_id,
            npc_id,
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
        })
    }

    pub fn predict(&self, text: &str, past_time: i64) -> Result<EmotionPrediction, EmotionPredictorError> {
        let mut emotion_predictor = EmotionPredictor::new()?;

        let predicted_emotion = emotion_predictor.predict_emotion(text).map_err(|e| {
            eprintln!("Error predicting emotion: {:?}", e);
            EmotionPredictorError::Inference(e.to_string())
        })?;

        self.evaluate_with_predicted_emotion(text, &predicted_emotion, past_time, None)
    }

    pub fn evaluate_with_predicted_emotion(
        &self,
        text: &str,
        predicted_emotion: &EmotionPrediction,
        past_time: i64,
        source_id: Option<&str>,
    ) -> Result<EmotionPrediction, EmotionPredictorError> {
        let global_emotion = self.calculate_current_emotion()?;

        let source_emotion = if let Some(source_id) = source_id.or(self.source_id.as_deref()) {
            Some(self.calculate_current_emotion_towards_source(source_id)?)
        } else {
            None
        };

        let final_emotion =
            self.combine_emotions_psychologically(predicted_emotion, source_emotion.as_ref(), &global_emotion);

        self.store_emotion_in_memory(text, &final_emotion, past_time, source_id)?;

        Ok(final_emotion)
    }

    pub fn calculate_current_emotion_towards_source(
        &self,
        source_id: &str,
    ) -> Result<EmotionPrediction, EmotionPredictorError> {
        let records = MemoryStore::get_by_source(&self.npc_id, source_id)
            .map_err(|e| EmotionPredictorError::Inference(format!("Failed to get memory records: {}", e)))?;

        let (valence, arousal) = self.calculate_weighted_emotion(&records);

        Ok(EmotionPrediction::new(valence, arousal))
    }

    pub fn calculate_current_emotion(&self) -> Result<EmotionPrediction, EmotionPredictorError> {
        let records = MemoryStore::get_all(&self.npc_id)
            .map_err(|e| EmotionPredictorError::Inference(format!("Failed to get memory records: {}", e)))?;

        if records.is_empty() {
            return Ok(EmotionPrediction::new(
                self.config.personality.valence,
                self.config.personality.arousal,
            ));
        }

        let (valence, arousal) = self.calculate_weighted_emotion(&records);

        Ok(EmotionPrediction::new(valence, arousal))
    }

    fn calculate_weighted_emotion(&self, records: &[MemoryRecord]) -> (f32, f32) {
        let decay_rate = self.config.memory.decay_rate;

        let personality_valence = self.config.personality.valence;
        let personality_arousal = self.config.personality.arousal;

        let mut weighted_valence = 0.0;
        let mut weighted_arousal = 0.0;
        let mut total_weight = 0.0;

        for record in records {
            let weight = E.powf(-decay_rate * record.past_time as f32);

            let valence_deviation = record.valence - personality_valence;
            let arousal_deviation = record.arousal - personality_arousal;

            weighted_valence += valence_deviation * weight;
            weighted_arousal += arousal_deviation * weight;
            total_weight += weight;
        }

        let final_valence = if total_weight > 0.0 {
            ((weighted_valence / total_weight) + personality_valence).clamp(-1.0, 1.0)
        } else {
            personality_valence
        };

        let final_arousal = if total_weight > 0.0 {
            ((weighted_arousal / total_weight) + personality_arousal).clamp(-1.0, 1.0)
        } else {
            personality_arousal
        };

        (final_valence, final_arousal)
    }

    pub fn combine_emotions_psychologically(
        &self,
        text_emotion: &EmotionPrediction,
        source_emotion: Option<&EmotionPrediction>,
        global_emotion: &EmotionPrediction,
    ) -> EmotionPrediction {
        let (final_valence, final_arousal) = if let Some(source_emotion) = source_emotion {
            let valence =
                (source_emotion.valence * 0.5) + (text_emotion.valence * 0.35) + (global_emotion.valence * 0.15);

            let arousal =
                (source_emotion.arousal * 0.5) + (text_emotion.arousal * 0.35) + (global_emotion.arousal * 0.15);

            (valence, arousal)
        } else {
            let valence = (text_emotion.valence * 0.7) + (global_emotion.valence * 0.3);
            let arousal = (text_emotion.arousal * 0.7) + (global_emotion.arousal * 0.3);

            (valence, arousal)
        };

        EmotionPrediction::new(final_valence.clamp(-1.0, 1.0), final_arousal.clamp(-1.0, 1.0))
    }

    fn store_emotion_in_memory(
        &self,
        text: &str,
        final_emotion: &EmotionPrediction,
        past_time: i64,
        source_id: Option<&str>,
    ) -> Result<(), EmotionPredictorError> {
        let effective_source_id = source_id.or(self.source_id.as_deref()).unwrap_or("unknown");

        let record = MemoryRecord {
            id: Uuid::new_v4().to_string(),
            source_id: effective_source_id.to_string(),
            text: text.to_string(),
            valence: final_emotion.valence,
            arousal: final_emotion.arousal,
            past_time,
        };

        MemoryStore::insert(&self.npc_id, record)
            .map_err(|e| EmotionPredictorError::Inference(format!("Failed to store memory: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::MemoryEmotionEvaluator;
    use crate::_test_mock::memory_mock::{MemoryTestHelpers, MockMemoryEmotionEvaluator, TestMemoryData};
    use crate::{EmotionPrediction, MemoryStore, NpcConfig};

    #[test]
    fn test_mock_memory_emotion_evaluator_new() {
        let config = NpcConfig::default();
        let evaluator = MockMemoryEmotionEvaluator::new(config, Some("test-source".to_string()));
        assert!(evaluator.is_ok());

        let evaluator = evaluator.unwrap();
        assert_eq!(evaluator.source_id, Some("test-source".to_string()));
        assert_eq!(evaluator.npc_id, "test-npc-id");
    }

    #[test]
    fn test_mock_memory_emotion_evaluator_new_no_source() {
        let config = NpcConfig::default();
        let evaluator = MockMemoryEmotionEvaluator::new(config, None);
        assert!(evaluator.is_ok());

        let evaluator = evaluator.unwrap();
        assert_eq!(evaluator.source_id, None);
    }

    #[test]
    fn test_mock_memory_emotion_evaluator_new_with_id() {
        let config = NpcConfig::default();
        let custom_id = "custom-npc-123".to_string();
        let evaluator =
            MockMemoryEmotionEvaluator::new_with_id(config, Some("test-source".to_string()), custom_id.clone());
        assert!(evaluator.is_ok());

        let evaluator = evaluator.unwrap();
        assert_eq!(evaluator.npc_id, custom_id);
    }

    #[test]
    fn test_mock_memory_emotion_evaluator_predict_default() {
        let evaluator = MockMemoryEmotionEvaluator::neutral();
        MemoryStore::clear(&evaluator.npc_id).unwrap();

        let result = evaluator.predict("Test message", 1000);

        assert!(result.is_ok());
        let prediction = result.unwrap();
        assert_eq!(prediction.valence, 0.0);
        assert_eq!(prediction.arousal, 0.0);

        assert_eq!(MemoryStore::get_memory_count(&evaluator.npc_id).unwrap(), 1);
    }

    #[test]
    fn test_mock_memory_emotion_evaluator_predict_custom_response() {
        let expected_emotion = EmotionPrediction::new(0.7, 0.5);
        let evaluator =
            MockMemoryEmotionEvaluator::neutral().with_predict_response("Happy message", Ok(expected_emotion.clone()));

        MemoryStore::clear(&evaluator.npc_id).unwrap();

        let result = evaluator.predict("Happy message", 1000);

        assert!(result.is_ok());
        let prediction = result.unwrap();
        assert_eq!(prediction.valence, expected_emotion.valence);
        assert_eq!(prediction.arousal, expected_emotion.arousal);
    }

    #[test]
    fn test_mock_memory_emotion_evaluator_predict_error() {
        let evaluator = MockMemoryEmotionEvaluator::error();
        let result = evaluator.predict("Test message", 1000);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Mock prediction error"));
    }

    #[test]
    fn test_mock_memory_emotion_evaluator_current_emotion() {
        let expected_emotion = EmotionPrediction::new(0.6, 0.4);
        let evaluator =
            MockMemoryEmotionEvaluator::neutral().with_current_emotion_response(Ok(expected_emotion.clone()));

        let result = evaluator.calculate_current_emotion();

        assert!(result.is_ok());
        let prediction = result.unwrap();
        assert_eq!(prediction.valence, expected_emotion.valence);
        assert_eq!(prediction.arousal, expected_emotion.arousal);
    }

    #[test]
    fn test_mock_memory_emotion_evaluator_source_emotion() {
        let expected_emotion = EmotionPrediction::new(0.8, 0.3);
        let evaluator =
            MockMemoryEmotionEvaluator::neutral().with_source_emotion_response("friend", Ok(expected_emotion.clone()));

        let result = evaluator.calculate_current_emotion_towards_source("friend");

        assert!(result.is_ok());
        let prediction = result.unwrap();
        assert_eq!(prediction.valence, expected_emotion.valence);
        assert_eq!(prediction.arousal, expected_emotion.arousal);
    }

    #[test]
    fn test_mock_memory_emotion_evaluator_clone() {
        let config = NpcConfig::default();
        let evaluator = MockMemoryEmotionEvaluator::new(config, Some("test-source".to_string())).unwrap();
        let cloned = evaluator.clone();

        assert_eq!(evaluator.source_id, cloned.source_id);
        assert_eq!(evaluator.npc_id, cloned.npc_id);
    }

    #[test]
    fn test_mock_memory_emotion_evaluator_presets() {
        let positive = MockMemoryEmotionEvaluator::positive();
        assert!(positive.source_id.is_some());

        let negative = MockMemoryEmotionEvaluator::negative();
        assert!(negative.source_id.is_some());

        let neutral = MockMemoryEmotionEvaluator::neutral();
        assert!(neutral.source_id.is_some());

        let error = MockMemoryEmotionEvaluator::error();
        assert!(error.source_id.is_some());
    }

    #[test]
    fn test_memory_test_helpers() {
        let npc_id = MemoryTestHelpers::test_npc_id();
        assert!(!npc_id.is_empty());

        let npc_ids = MemoryTestHelpers::test_npc_ids(3);
        assert_eq!(npc_ids.len(), 3);
        assert_ne!(npc_ids[0], npc_ids[1]);
        assert_ne!(npc_ids[1], npc_ids[2]);

        let result = MemoryTestHelpers::setup_test_memories(&npc_id);
        assert!(result.is_ok());

        let count = MemoryStore::get_memory_count(&npc_id).unwrap();
        assert!(count > 0);

        let clear_result = MemoryTestHelpers::clear_test_npc_memory(&npc_id);
        assert!(clear_result.is_ok());
        assert_eq!(MemoryStore::get_memory_count(&npc_id).unwrap(), 0);
    }

    #[test]
    fn test_memory_record_validation() {
        let valid_record =
            MemoryTestHelpers::create_valid_memory_record("test-1", "source-1", "Test message", 0.5, -0.3, 1000);
        assert!(valid_record.is_ok());

        let clamped_record =
            MemoryTestHelpers::create_valid_memory_record("test-2", "source-2", "Test message", 2.0, -2.0, 2000);
        assert!(clamped_record.is_ok());
        let record = clamped_record.unwrap();
        assert_eq!(record.valence, 1.0);
        assert_eq!(record.arousal, -1.0);
    }

    #[test]
    fn test_test_memory_data() {
        let sample_records = TestMemoryData::sample_memory_records();
        assert_eq!(sample_records.len(), 4);

        for record in &sample_records {
            let validation = MemoryTestHelpers::validate_memory_record(record);
            assert!(
                validation.is_ok(),
                "Record {} failed validation: {:?}",
                record.id,
                validation
            );
        }

        let happy_record = TestMemoryData::happy_memory_record();
        assert!(happy_record.valence > 0.0);
        assert!(MemoryTestHelpers::validate_memory_record(&happy_record).is_ok());

        let sad_record = TestMemoryData::sad_memory_record();
        assert!(sad_record.valence < 0.0);
        assert!(MemoryTestHelpers::validate_memory_record(&sad_record).is_ok());

        let angry_record = TestMemoryData::angry_memory_record();
        assert!(angry_record.valence < 0.0);
        assert!(angry_record.arousal > 0.0);
        assert!(MemoryTestHelpers::validate_memory_record(&angry_record).is_ok());

        let neutral_record = TestMemoryData::neutral_memory_record();
        assert_eq!(neutral_record.valence, 0.0);
        assert_eq!(neutral_record.arousal, 0.0);
        assert!(MemoryTestHelpers::validate_memory_record(&neutral_record).is_ok());
    }

    #[test]
    fn test_real_memory_emotion_evaluator_new() {
        let config = NpcConfig::default();
        let evaluator = MemoryEmotionEvaluator::new(config, Some("test-source".to_string()));
        assert!(evaluator.is_ok());

        let evaluator = evaluator.unwrap();
        assert_eq!(evaluator.source_id, Some("test-source".to_string()));
        assert!(!evaluator.npc_id.is_empty());
    }

    #[test]
    fn test_real_memory_emotion_evaluator_new_no_source() {
        let config = NpcConfig::default();
        let evaluator = MemoryEmotionEvaluator::new(config, None);
        assert!(evaluator.is_ok());

        let evaluator = evaluator.unwrap();
        assert_eq!(evaluator.source_id, None);
    }

    #[test]
    fn test_real_memory_emotion_evaluator_new_with_id() {
        let config = NpcConfig::default();
        let custom_id = "custom-npc-456".to_string();
        let evaluator = MemoryEmotionEvaluator::new_with_id(config, Some("test-source".to_string()), custom_id.clone());
        assert!(evaluator.is_ok());

        let evaluator = evaluator.unwrap();
        assert_eq!(evaluator.npc_id, custom_id);
    }

    #[test]
    fn test_real_memory_emotion_evaluator_clone() {
        let config = NpcConfig::default();
        let evaluator = MemoryEmotionEvaluator::new(config, Some("test-source".to_string())).unwrap();
        let cloned = evaluator.clone();

        assert_eq!(evaluator.source_id, cloned.source_id);
        assert_eq!(evaluator.npc_id, cloned.npc_id);
    }

    #[test]
    fn test_real_memory_emotion_evaluator_calculate_current_emotion_empty() {
        let config = NpcConfig::default();
        let evaluator = MemoryEmotionEvaluator::new(config, Some("test-source".to_string())).unwrap();

        MemoryStore::clear(&evaluator.npc_id).unwrap();

        let result = evaluator.calculate_current_emotion();
        assert!(result.is_ok());

        let emotion = result.unwrap();
        assert_eq!(emotion.valence, evaluator.config.personality.valence);
        assert_eq!(emotion.arousal, evaluator.config.personality.arousal);
    }
}
