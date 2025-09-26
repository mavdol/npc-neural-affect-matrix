pub mod mock;

use npc_neural_affect_matrix::modules::memory::store::{MemoryStore, MemoryRecord};
use npc_neural_affect_matrix::modules::memory::evaluator::MemoryEmotionEvaluator;
use npc_neural_affect_matrix::config::NpcConfig;
use npc_neural_affect_matrix::EmotionPrediction;


use mock::{MockMemoryEmotionEvaluator, TestMemoryData, MemoryTestHelpers};

fn test_npc_id() -> String {
    format!("test-npc-{}", uuid::Uuid::new_v4())
}

#[test]
fn test_memory_store_new() {
    let store = MemoryStore::new();
    assert!(matches!(store, MemoryStore));
}

#[test]
fn test_memory_store_default() {
    let store = MemoryStore::default();
    assert!(matches!(store, MemoryStore));
}

#[test]
fn test_memory_store_insert() {
    let npc_id = test_npc_id();
    MemoryStore::clear(&npc_id).unwrap();

    let record = MemoryRecord {
        id: "test-1".to_string(),
        source_id: "source-1".to_string(),
        text: "Test message".to_string(),
        valence: 0.3,
        arousal: -0.2,
        past_time: 1000,
    };

    let result = MemoryStore::insert(&npc_id, record);
    assert!(result.is_ok());

    let count = MemoryStore::get_memory_count(&npc_id).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_memory_store_get_all() {
    let npc_id = test_npc_id();
    MemoryStore::clear(&npc_id).unwrap();

    let record1 = MemoryRecord {
        id: "test-1".to_string(),
        source_id: "source-1".to_string(),
        text: "Message 1".to_string(),
        valence: 0.5,
        arousal: -0.3,
        past_time: 1000,
    };

    let record2 = MemoryRecord {
        id: "test-2".to_string(),
        source_id: "source-2".to_string(),
        text: "Message 2".to_string(),
        valence: -0.2,
        arousal: 0.7,
        past_time: 2000,
    };

    MemoryStore::insert(&npc_id, record1).unwrap();
    MemoryStore::insert(&npc_id, record2).unwrap();

    let all_records = MemoryStore::get_all(&npc_id).unwrap();
    assert_eq!(all_records.len(), 2);
    assert_eq!(all_records[0].id, "test-1");
    assert_eq!(all_records[1].id, "test-2");
}

#[test]
fn test_memory_store_get_by_source() {
    let npc_id = test_npc_id();
    MemoryStore::clear(&npc_id).unwrap();

    let record1 = MemoryRecord {
        id: "test-1".to_string(),
        source_id: "source-1".to_string(),
        text: "Message 1".to_string(),
        valence: 0.5,
        arousal: -0.3,
        past_time: 1000,
    };

    let record2 = MemoryRecord {
        id: "test-2".to_string(),
        source_id: "source-2".to_string(),
        text: "Message 2".to_string(),
        valence: -0.2,
        arousal: 0.7,
        past_time: 2000,
    };

    MemoryStore::insert(&npc_id, record1).unwrap();
    MemoryStore::insert(&npc_id, record2).unwrap();

    let source1_records = MemoryStore::get_by_source(&npc_id, "source-1").unwrap();
    assert_eq!(source1_records.len(), 1);
    assert_eq!(source1_records[0].id, "test-1");

    let source2_records = MemoryStore::get_by_source(&npc_id, "source-2").unwrap();
    assert_eq!(source2_records.len(), 1);
    assert_eq!(source2_records[0].id, "test-2");

    let unknown_records = MemoryStore::get_by_source(&npc_id, "unknown").unwrap();
    assert_eq!(unknown_records.len(), 0);
}

#[test]
fn test_memory_store_import_valid() {
    let npc_id = test_npc_id();
    MemoryStore::clear(&npc_id).unwrap();

    let records = vec![
        MemoryRecord {
            id: "import-1".to_string(),
            source_id: "source-1".to_string(),
            text: "Imported message 1".to_string(),
            valence: 0.8,
            arousal: -0.4,
            past_time: 1000,
        },
        MemoryRecord {
            id: "import-2".to_string(),
            source_id: "source-2".to_string(),
            text: "Imported message 2".to_string(),
            valence: -0.6,
            arousal: 0.9,
            past_time: 2000,
        },
    ];

    let result = MemoryStore::import(&npc_id, records);
    assert!(result.is_ok());

    let all_records = MemoryStore::get_all(&npc_id).unwrap();
    assert_eq!(all_records.len(), 2);
    assert_eq!(all_records[0].id, "import-1");
    assert_eq!(all_records[1].id, "import-2");
}

#[test]
fn test_memory_store_import_empty_id() {
    let npc_id = test_npc_id();
    MemoryStore::clear(&npc_id).unwrap();

    let records = vec![
        MemoryRecord {
            id: "".to_string(),
            source_id: "source-1".to_string(),
            text: "Invalid message".to_string(),
            valence: 0.5,
            arousal: -0.3,
            past_time: 1000,
        },
    ];

    let result = MemoryStore::import(&npc_id, records);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("empty ID"));
}

#[test]
fn test_memory_store_import_invalid_valence() {
    let npc_id = test_npc_id();
    MemoryStore::clear(&npc_id).unwrap();

    let records = vec![
        MemoryRecord {
            id: "test-1".to_string(),
            source_id: "source-1".to_string(),
            text: "Invalid valence".to_string(),
            valence: 1.5,
            arousal: -0.3,
            past_time: 1000,
        },
    ];

    let result = MemoryStore::import(&npc_id, records);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("invalid valence"));
}

#[test]
fn test_memory_store_import_invalid_arousal() {
    let npc_id = test_npc_id();
    MemoryStore::clear(&npc_id).unwrap();

    let records = vec![
        MemoryRecord {
            id: "test-1".to_string(),
            source_id: "source-1".to_string(),
            text: "Invalid arousal".to_string(),
            valence: 0.5,
            arousal: -1.5,
            past_time: 1000,
        },
    ];

    let result = MemoryStore::import(&npc_id, records);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("invalid arousal"));
}

#[test]
fn test_memory_store_clear() {
    let npc_id = test_npc_id();
    MemoryStore::clear(&npc_id).unwrap();

    let record = MemoryRecord {
        id: "test-1".to_string(),
        source_id: "source-1".to_string(),
        text: "Test message".to_string(),
        valence: 0.5,
        arousal: -0.3,
        past_time: 1000,
    };

    MemoryStore::insert(&npc_id, record).unwrap();
    assert_eq!(MemoryStore::get_memory_count(&npc_id).unwrap(), 1);

    MemoryStore::clear(&npc_id).unwrap();
    assert_eq!(MemoryStore::get_memory_count(&npc_id).unwrap(), 0);
}

#[test]
fn test_memory_store_get_memory_count() {
    let npc_id = test_npc_id();
    MemoryStore::clear(&npc_id).unwrap();
    assert_eq!(MemoryStore::get_memory_count(&npc_id).unwrap(), 0);

    let record1 = MemoryRecord {
        id: "test-1".to_string(),
        source_id: "source-1".to_string(),
        text: "Message 1".to_string(),
        valence: 0.5,
        arousal: -0.3,
        past_time: 1000,
    };

    let record2 = MemoryRecord {
        id: "test-2".to_string(),
        source_id: "source-2".to_string(),
        text: "Message 2".to_string(),
        valence: -0.2,
        arousal: 0.7,
        past_time: 2000,
    };

    MemoryStore::insert(&npc_id, record1).unwrap();
    assert_eq!(MemoryStore::get_memory_count(&npc_id).unwrap(), 1);

    MemoryStore::insert(&npc_id, record2).unwrap();
    assert_eq!(MemoryStore::get_memory_count(&npc_id).unwrap(), 2);
}

#[test]
fn test_memory_store_import_boundary_values() {
    let npc_id = test_npc_id();
    MemoryStore::clear(&npc_id).unwrap();

    let records = vec![
        MemoryRecord {
            id: "boundary-1".to_string(),
            source_id: "source-1".to_string(),
            text: "Max positive".to_string(),
            valence: 1.0,
            arousal: 1.0,
            past_time: 1000,
        },
        MemoryRecord {
            id: "boundary-2".to_string(),
            source_id: "source-2".to_string(),
            text: "Max negative".to_string(),
            valence: -1.0,
            arousal: -1.0,
            past_time: 2000,
        },
        MemoryRecord {
            id: "boundary-3".to_string(),
            source_id: "source-3".to_string(),
            text: "Zero values".to_string(),
            valence: 0.0,
            arousal: 0.0,
            past_time: 3000,
        },
    ];

    let result = MemoryStore::import(&npc_id, records);
    assert!(result.is_ok());

    let all_records = MemoryStore::get_all(&npc_id).unwrap();
    assert_eq!(all_records.len(), 3);
}

#[test]
fn test_memory_store_multiple_npcs() {
    let npc_id1 = "npc-1".to_string();
    let npc_id2 = "npc-2".to_string();

    MemoryStore::clear(&npc_id1).unwrap();
    MemoryStore::clear(&npc_id2).unwrap();

    let record1 = MemoryRecord {
        id: "test-1".to_string(),
        source_id: "source-1".to_string(),
        text: "NPC 1 message".to_string(),
        valence: 0.5,
        arousal: -0.3,
        past_time: 1000,
    };

    let record2 = MemoryRecord {
        id: "test-2".to_string(),
        source_id: "source-2".to_string(),
        text: "NPC 2 message".to_string(),
        valence: -0.2,
        arousal: 0.7,
        past_time: 2000,
    };

    MemoryStore::insert(&npc_id1, record1).unwrap();
    MemoryStore::insert(&npc_id2, record2).unwrap();

    assert_eq!(MemoryStore::get_memory_count(&npc_id1).unwrap(), 1);
    assert_eq!(MemoryStore::get_memory_count(&npc_id2).unwrap(), 1);

    let npc1_records = MemoryStore::get_all(&npc_id1).unwrap();
    let npc2_records = MemoryStore::get_all(&npc_id2).unwrap();

    assert_eq!(npc1_records.len(), 1);
    assert_eq!(npc2_records.len(), 1);
    assert_eq!(npc1_records[0].text, "NPC 1 message");
    assert_eq!(npc2_records[0].text, "NPC 2 message");
}

#[test]
fn test_memory_store_remove_npc() {
    let npc_id = test_npc_id();
    MemoryStore::clear(&npc_id).unwrap();

    let record = MemoryRecord {
        id: "test-1".to_string(),
        source_id: "source-1".to_string(),
        text: "Test message".to_string(),
        valence: 0.5,
        arousal: -0.3,
        past_time: 1000,
    };

    MemoryStore::insert(&npc_id, record).unwrap();
    assert_eq!(MemoryStore::get_memory_count(&npc_id).unwrap(), 1);

    MemoryStore::remove_npc(&npc_id).unwrap();
    assert_eq!(MemoryStore::get_memory_count(&npc_id).unwrap(), 0);
}

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
    let evaluator = MockMemoryEmotionEvaluator::new_with_id(config, Some("test-source".to_string()), custom_id.clone());
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
    let evaluator = MockMemoryEmotionEvaluator::neutral()
        .with_predict_response("Happy message", Ok(expected_emotion.clone()));

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
    let evaluator = MockMemoryEmotionEvaluator::neutral()
        .with_current_emotion_response(Ok(expected_emotion.clone()));

    let result = evaluator.calculate_current_emotion();

    assert!(result.is_ok());
    let prediction = result.unwrap();
    assert_eq!(prediction.valence, expected_emotion.valence);
    assert_eq!(prediction.arousal, expected_emotion.arousal);
}

#[test]
fn test_mock_memory_emotion_evaluator_source_emotion() {
    let expected_emotion = EmotionPrediction::new(0.8, 0.3);
    let evaluator = MockMemoryEmotionEvaluator::neutral()
        .with_source_emotion_response("friend", Ok(expected_emotion.clone()));

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
    let valid_record = MemoryTestHelpers::create_valid_memory_record(
        "test-1",
        "source-1",
        "Test message",
        0.5,
        -0.3,
        1000,
    );
    assert!(valid_record.is_ok());

    let clamped_record = MemoryTestHelpers::create_valid_memory_record(
        "test-2",
        "source-2",
        "Test message",
        2.0,
        -2.0,
        2000,
    );
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
        assert!(validation.is_ok(), "Record {} failed validation: {:?}", record.id, validation);
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
