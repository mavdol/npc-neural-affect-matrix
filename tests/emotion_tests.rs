pub mod mock;

use npc_neural_affect_matrix::{EmotionPrediction, EmotionPredictorError, EmotionPredictor};
use std::path::Path;

use mock::{
    MockEmotionPredictor, TestEmotionData, EmotionPredict, create_mock_model_directory
};


#[test]
fn test_io_error_conversion() {
    let io_error = std::fs::read_to_string("/nonexistent/path/file.txt").unwrap_err();
    let emotion_error: EmotionPredictorError = io_error.into();
    assert!(matches!(emotion_error, EmotionPredictorError::Io(_)));
}

#[test]
fn test_tokenizer_error() {
    let mut mock_predictor = MockEmotionPredictor::new()
        .with_response("invalid_tokenizer_input", Err(EmotionPredictorError::Tokenizer("Invalid tokenizer input".to_string())));

    let result = mock_predictor.predict_emotion("invalid_tokenizer_input");
    match result {
        Err(EmotionPredictorError::Tokenizer(_)) => {}
        Err(e) => panic!("Expected Tokenizer error, got: {:?}", e),
        Ok(_) => panic!("Expected error, but got Ok"),
    }
}

#[test]
fn test_model_loading_error() {
    let mut mock_predictor = MockEmotionPredictor::new()
        .with_response("model_loading_test", Err(EmotionPredictorError::ModelLoading("Failed to load model".to_string())));

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
        .with_response("Error text", Err(EmotionPredictorError::Inference("Test error".to_string())));

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
        },
        Err(e) => {
            assert!(matches!(e, EmotionPredictorError::ModelLoading(_) | EmotionPredictorError::Io(_)));
        }
    }
}

#[test]
fn test_load_tokenizer_with_fallback_existing_file() {
    if let Ok(temp_dir) = create_mock_model_directory() {
        let tokenizer_path = temp_dir.join("tokenizer.json");

        let result = EmotionPredictor::load_tokenizer_with_fallback(&tokenizer_path);

        let _ = std::fs::remove_dir_all(&temp_dir);

        assert!(result.is_ok(), "Expected tokenizer to load successfully, but got error: {:?}", result);
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
