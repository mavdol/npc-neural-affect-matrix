use std::os::raw::c_char;

use crate::api::types::{ApiResult, NpcId};
use crate::api::services::validation_service::parse_c_string;
use crate::modules::memory::store::{MemoryRecord, MemoryStore};

pub fn import_memory(npc_id: &NpcId, npc_memory_json: *const c_char) -> Result<(), *mut ApiResult> {
    let memory_str = match parse_c_string(npc_memory_json, "Memory string") {
        Ok(s) => s,
        Err(result) => return Err(result),
    };

    if memory_str.is_empty() {
        return Ok(());
    }

    let memory_records: Vec<MemoryRecord> =
        match serde_json::from_str(&memory_str) {
            Ok(records) => records,
            Err(e) => return Err(Box::into_raw(Box::new(ApiResult::error(format!("Failed to parse memory: {}", e))))),
        };

    MemoryStore::import(npc_id, memory_records)
        .map_err(|e| Box::into_raw(Box::new(ApiResult::error(format!("Failed to import memory: {}", e)))))
}

pub fn get_all_memory(npc_id: &NpcId) -> Result<String, *mut ApiResult> {
    let memory_records = match MemoryStore::get_all(npc_id) {
        Ok(records) => records,
        Err(e) => return Err(Box::into_raw(Box::new(ApiResult::error(format!("Failed to get memory records: {}", e))))),
    };

    match serde_json::to_string(&memory_records) {
        Ok(json) => Ok(json),
        Err(e) => Err(Box::into_raw(Box::new(ApiResult::error(format!("Failed to serialize memory: {}", e))))),
    }
}

pub fn clear_memory(npc_id: &NpcId) -> Result<String, *mut ApiResult> {
    match MemoryStore::clear(npc_id) {
        Ok(_) => Ok("Memory cleared successfully".to_string()),
        Err(e) => Err(Box::into_raw(Box::new(ApiResult::error(format!("Failed to clear memory: {}", e))))),
    }
}
