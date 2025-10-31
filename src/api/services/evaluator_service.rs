use crate::api::types::{ApiResult, NpcId};
use crate::{EmotionPrediction, EmotionPredictor, MemoryEmotionEvaluator};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub static NPC_SESSIONS: OnceLock<Mutex<HashMap<NpcId, MemoryEmotionEvaluator>>> = OnceLock::new();
pub static SHARED_MODEL: OnceLock<Arc<Mutex<EmotionPredictor>>> = OnceLock::new();

pub fn initialize_shared_model() -> Result<(), *mut ApiResult> {
    let predictor = EmotionPredictor::new().map_err(|e| {
        Box::into_raw(Box::new(ApiResult::error(format!(
            "Failed to initialize model: {:?}",
            e
        ))))
    })?;

    SHARED_MODEL
        .set(Arc::new(Mutex::new(predictor)))
        .map_err(|_| Box::into_raw(Box::new(ApiResult::error("Model already initialized".to_string()))))?;

    Ok(())
}

pub fn get_npc_sessions_with_timeout(
) -> Result<std::sync::MutexGuard<'static, HashMap<NpcId, MemoryEmotionEvaluator>>, *mut ApiResult> {
    let sessions_mutex = NPC_SESSIONS.get_or_init(|| Mutex::new(HashMap::new()));

    sessions_mutex.lock().map_err(|_| {
        Box::into_raw(Box::new(ApiResult::error(
            "Failed to acquire session lock - mutex poisoned".to_string(),
        )))
    })
}

pub fn get_npc_sessions(
) -> Result<std::sync::MutexGuard<'static, HashMap<NpcId, MemoryEmotionEvaluator>>, *mut ApiResult> {
    get_npc_sessions_with_timeout()
}

pub fn create_npc_session(npc_id: NpcId, evaluator: MemoryEmotionEvaluator) -> Result<(), *mut ApiResult> {
    let mut sessions = get_npc_sessions()?;
    if sessions.contains_key(&npc_id) {
        return Err(Box::into_raw(Box::new(ApiResult::error(format!(
            "NPC session '{}' already exists",
            npc_id
        )))));
    }
    sessions.insert(npc_id, evaluator);
    Ok(())
}

pub fn remove_npc_session(npc_id: &NpcId) -> Result<(), *mut ApiResult> {
    let mut sessions = get_npc_sessions()?;
    if sessions.remove(npc_id).is_none() {
        return Err(Box::into_raw(Box::new(ApiResult::error(format!(
            "NPC session '{}' not found",
            npc_id
        )))));
    }
    Ok(())
}

pub fn with_npc_evaluator<F>(npc_id: &NpcId, f: F) -> *mut ApiResult
where
    F: FnOnce(&MemoryEmotionEvaluator) -> Result<String, String>,
{
    let sessions = match get_npc_sessions() {
        Ok(sessions) => sessions,
        Err(result) => return result,
    };

    let evaluator = match sessions.get(npc_id) {
        Some(e) => e,
        None => {
            return Box::into_raw(Box::new(ApiResult::error(format!(
                "NPC session '{}' not found. Call create_npc_session first.",
                npc_id
            ))))
        }
    };

    match f(evaluator) {
        Ok(data) => Box::into_raw(Box::new(ApiResult::success(data))),
        Err(error) => Box::into_raw(Box::new(ApiResult::error(error))),
    }
}

pub fn create_working_evaluator(
    evaluator: &MemoryEmotionEvaluator,
    source_id: Option<&str>,
) -> Result<MemoryEmotionEvaluator, *mut ApiResult> {
    match source_id {
        Some(source) => MemoryEmotionEvaluator::new(evaluator.config.clone(), Some(source.to_string())).map_err(|e| {
            Box::into_raw(Box::new(ApiResult::error(format!(
                "Failed to create source evaluator: {:?}",
                e
            ))))
        }),
        None => Ok(evaluator.clone()),
    }
}

pub fn predict_with_cached_model(text: &str) -> Result<EmotionPrediction, *mut ApiResult> {
    let model_arc = SHARED_MODEL.get().ok_or_else(|| {
        Box::into_raw(Box::new(ApiResult::error(
            "Model not initialized. Call initialize_neural_matrix first.".to_string(),
        )))
    })?;

    let mut model = model_arc
        .lock()
        .map_err(|_| Box::into_raw(Box::new(ApiResult::error("Failed to acquire model lock".to_string()))))?;

    model
        .predict_emotion(text)
        .map_err(|e| Box::into_raw(Box::new(ApiResult::error(format!("Prediction failed: {:?}", e)))))
}

pub fn format_emotion_json(emotion: &EmotionPrediction) -> String {
    serde_json::json!({
        "valence": emotion.valence,
        "arousal": emotion.arousal
    })
    .to_string()
}

pub fn evaluate_interaction_with_cached_model(
    evaluator: &MemoryEmotionEvaluator,
    text: &str,
    source_id: Option<&str>,
) -> Result<EmotionPrediction, String> {
    let predicted_emotion =
        predict_with_cached_model(text).map_err(|_| "Failed to predict emotion with cached model".to_string())?;

    evaluator
        .evaluate_with_predicted_emotion(text, &predicted_emotion, 0, source_id)
        .map_err(|e| format!("Failed to evaluate interaction: {:?}", e))
}
