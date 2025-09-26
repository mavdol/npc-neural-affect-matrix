use npc_neural_affect_matrix::{EmotionPrediction, EmotionPredictorError};
use std::collections::HashMap;

pub struct MockEmotionPredictor {
    responses: HashMap<String, Result<EmotionPrediction, EmotionPredictorError>>,
    default_response: Result<EmotionPrediction, EmotionPredictorError>,
}

impl MockEmotionPredictor {
    pub fn new() -> Self {
        Self {
            responses: HashMap::new(),
            default_response: Ok(EmotionPrediction::new(0.0, 0.0)),
        }
    }

    pub fn with_response(mut self, text: &str, response: Result<EmotionPrediction, EmotionPredictorError>) -> Self {
        self.responses.insert(text.to_string(), response);
        self
    }

    pub fn with_default_response(mut self, response: Result<EmotionPrediction, EmotionPredictorError>) -> Self {
        self.default_response = response;
        self
    }

    pub fn positive() -> Self {
        Self::new().with_default_response(Ok(EmotionPrediction::new(0.8, 0.6)))
    }

    pub fn negative() -> Self {
        Self::new().with_default_response(Ok(EmotionPrediction::new(-0.7, -0.4)))
    }

    pub fn neutral() -> Self {
        Self::new().with_default_response(Ok(EmotionPrediction::new(0.0, 0.0)))
    }

    pub fn error() -> Self {
        Self::new().with_default_response(Err(EmotionPredictorError::Inference("Mock error".to_string())))
    }
}

impl EmotionPredict for MockEmotionPredictor {
    fn predict_emotion(&mut self, text: &str) -> Result<EmotionPrediction, EmotionPredictorError> {
        self.responses.get(text).cloned().unwrap_or_else(|| self.default_response.clone())
    }
}

pub struct TestEmotionData;

impl TestEmotionData {
    pub fn happy_texts() -> Vec<&'static str> {
        vec![
            "I'm so happy today!",
            "This is wonderful news!",
            "I love spending time with friends",
            "What a beautiful day!",
        ]
    }

    pub fn sad_texts() -> Vec<&'static str> {
        vec![
            "I'm feeling really down",
            "This is terrible news",
            "I miss my old friends",
            "Everything seems so dark",
        ]
    }

    pub fn angry_texts() -> Vec<&'static str> {
        vec![
            "This makes me so angry!",
            "I can't believe this happened",
            "This is completely unfair",
            "I'm furious about this situation",
        ]
    }

    pub fn neutral_texts() -> Vec<&'static str> {
        vec![
            "The weather is okay today",
            "I went to the store",
            "The meeting is at 3 PM",
            "Please review the document",
        ]
    }

    pub fn expected_happy_prediction() -> EmotionPrediction {
        EmotionPrediction::new(0.75, 0.6)  // High valence, moderate arousal
    }

    pub fn expected_sad_prediction() -> EmotionPrediction {
        EmotionPrediction::new(-0.6, -0.3)  // Low valence, low arousal
    }

    pub fn expected_angry_prediction() -> EmotionPrediction {
        EmotionPrediction::new(-0.7, 0.8)  // Low valence, high arousal
    }

    pub fn expected_neutral_prediction() -> EmotionPrediction {
        EmotionPrediction::new(0.0, 0.0)  // Neutral valence and arousal
    }
}

pub fn create_mock_model_directory() -> Result<std::path::PathBuf, std::io::Error> {
    let temp_dir = std::env::temp_dir().join(format!("test_model_{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir)?;

    let tokenizer_content = format!(r#"{{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {{"id": 0, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
            {{"id": 1, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
            {{"id": 2, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
            {{"id": 3, "content": "[PAD]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}}
        ],
        "normalizer": {{
            "type": "BertNormalizer",
            "clean_text": true,
            "handle_chinese_chars": true,
            "strip_accents": null,
            "lowercase": true
        }},
        "pre_tokenizer": {{
            "type": "BertPreTokenizer"
        }},
        "post_processor": {{
            "type": "BertProcessing",
            "sep": ["[SEP]", 2],
            "cls": ["[CLS]", 1]
        }},
        "decoder": {{
            "type": "WordPiece",
            "prefix": "{}",
            "cleanup": true
        }},
        "model": {{
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "{}",
            "max_input_chars_per_word": 100,
            "vocab": {{
                "[UNK]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
                "[PAD]": 3,
                "hello": 4,
                "world": 5,
                "test": 6,
                "happy": 7,
                "sad": 8,
                "angry": 9,
                "neutral": 10
            }}
        }}
    }}"#, "##", "##");

    std::fs::write(temp_dir.join("tokenizer.json"), tokenizer_content)?;

    let mock_model_data = vec![0u8; 2000]; // Larger than the 1000 byte threshold
    std::fs::write(temp_dir.join("model.onnx"), mock_model_data)?;

    Ok(temp_dir)
}


pub trait EmotionPredict {
    fn predict_emotion(&mut self, text: &str) -> Result<EmotionPrediction, EmotionPredictorError>;
}

pub trait ModelLoader {
    fn load_model(&self, model_path: &str) -> Result<Box<dyn EmotionPredict>, EmotionPredictorError>;
}

pub trait TokenizerTrait {
    fn tokenize(&self, text: &str) -> Result<(Vec<u32>, Vec<u32>), EmotionPredictorError>;
}

pub trait ModelInference {
    fn run_inference(&self, input_ids: &[i64], attention_mask: &[i64]) -> Result<(f32, f32), EmotionPredictorError>;
}
