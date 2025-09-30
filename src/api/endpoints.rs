use std::os::raw::c_char;
use crate::{MemoryEmotionEvaluator, NpcConfig};
use crate::api::{
    types::ApiResult,
    services::{
        validation_service::*,
        evaluator_service::{create_npc_session as create_session, remove_npc_session as remove_session, with_npc_evaluator, format_emotion_json, initialize_shared_model, evaluate_interaction_with_cached_model},
        memory_service::{import_memory, get_all_memory, clear_memory}
    }
};
use crate::modules::memory::store::MemoryStore;

#[no_mangle]
pub extern "C" fn initialize_neural_matrix() -> *mut ApiResult {
    match initialize_shared_model() {
        Ok(()) => Box::into_raw(Box::new(ApiResult::success("Model initialized successfully".to_string()))),
        Err(result) => result,
    }
}

#[no_mangle]
pub extern "C" fn create_npc_session(config_json: *const c_char, npc_memory_json: *const c_char) -> *mut ApiResult {
    let npc_id = uuid::Uuid::new_v4().to_string();

    let config_str = match parse_c_string(config_json, "Config string") {
        Ok(s) => s,
        Err(result) => return result,
    };

    let config: NpcConfig = match serde_json::from_str(&config_str) {
        Ok(c) => c,
        Err(e) => return Box::into_raw(Box::new(ApiResult::error(format!("Failed to parse config: {}", e)))),
    };

    if !npc_memory_json.is_null() {
        if let Err(result) = import_memory(&npc_id, npc_memory_json) {
            return result;
        }
    }

    let evaluator = match MemoryEmotionEvaluator::new_with_id(config, None, npc_id.clone()) {
        Ok(e) => e,
        Err(e) => return Box::into_raw(Box::new(ApiResult::error(format!("Failed to create evaluator: {:?}", e)))),
    };

    if let Err(result) = create_session(npc_id.clone(), evaluator) {
        return result;
    }

    let response_data = serde_json::json!({
        "npc_id": npc_id
    }).to_string();

    Box::into_raw(Box::new(ApiResult::success(response_data)))
}

#[no_mangle]
pub extern "C" fn remove_npc_session(npc_id: *const c_char) -> *mut ApiResult {
    let npc_id_str = match parse_c_string(npc_id, "NPC ID string") {
        Ok(s) => s,
        Err(result) => return result,
    };

    if let Err(e) = MemoryStore::remove_npc(&npc_id_str) {
        return Box::into_raw(Box::new(ApiResult::error(format!("Failed to remove NPC memory: {}", e))));
    }

    if let Err(result) = remove_session(&npc_id_str) {
        return result;
    }

    Box::into_raw(Box::new(ApiResult::success(format!("NPC session '{}' removed successfully", npc_id_str))))
}


#[no_mangle]
pub extern "C" fn evaluate_interaction(npc_id: *const c_char, text: *const c_char, source_id: *const c_char) -> *mut ApiResult {
    let npc_id_str = match parse_c_string(npc_id, "NPC ID string") {
        Ok(s) => s,
        Err(result) => return result,
    };

    let text_str = match parse_c_string(text, "Text string") {
        Ok(s) => s,
        Err(result) => return result,
    };

    let source_str = parse_optional_c_string(source_id);

    with_npc_evaluator(&npc_id_str, |evaluator| {
        let final_emotion = evaluate_interaction_with_cached_model(
            evaluator,
            &text_str,
            source_str.as_deref()
        )?;

        Ok(format_emotion_json(&final_emotion))
    })
}


#[no_mangle]
pub extern "C" fn get_current_emotion(npc_id: *const c_char) -> *mut ApiResult {
    let npc_id_str = match parse_c_string(npc_id, "NPC ID string") {
        Ok(s) => s,
        Err(result) => return result,
    };

    with_npc_evaluator(&npc_id_str, |evaluator| {
        let emotion = evaluator.calculate_current_emotion()
            .map_err(|e| format!("Failed to calculate emotion: {:?}", e))?;
        Ok(format_emotion_json(&emotion))
    })
}

#[no_mangle]
pub extern "C" fn get_current_emotion_by_source_id(npc_id: *const c_char, source_id: *const c_char) -> *mut ApiResult {
    let npc_id_str = match parse_c_string(npc_id, "NPC ID string") {
        Ok(s) => s,
        Err(result) => return result,
    };

    let source_str = match parse_c_string(source_id, "Source ID string") {
        Ok(s) => s,
        Err(result) => return result,
    };

    with_npc_evaluator(&npc_id_str, |evaluator| {
        let emotion = evaluator.calculate_current_emotion_towards_source(&source_str)
            .map_err(|e| format!("Failed to calculate emotion towards source: {:?}", e))?;
        Ok(format_emotion_json(&emotion))
    })
}

#[no_mangle]
pub extern "C" fn get_npc_memory(npc_id: *const c_char) -> *mut ApiResult {
    let npc_id_str = match parse_c_string(npc_id, "NPC ID string") {
        Ok(s) => s,
        Err(result) => return result,
    };

    match get_all_memory(&npc_id_str) {
        Ok(json) => Box::into_raw(Box::new(ApiResult::success(json))),
        Err(result) => result,
    }
}

#[no_mangle]
pub extern "C" fn clear_npc_memory(npc_id: *const c_char) -> *mut ApiResult {
    let npc_id_str = match parse_c_string(npc_id, "NPC ID string") {
        Ok(s) => s,
        Err(result) => return result,
    };

    match clear_memory(&npc_id_str) {
        Ok(message) => Box::into_raw(Box::new(ApiResult::success(message))),
        Err(result) => result,
    }
}

#[no_mangle]
pub extern "C" fn free_api_result(result: *mut ApiResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        let result = Box::from_raw(result);
        if !result.data.is_null() {
            let _ = std::ffi::CString::from_raw(result.data);
        }
        if !result.error.is_null() {
            let _ = std::ffi::CString::from_raw(result.error);
        }
    }
}


