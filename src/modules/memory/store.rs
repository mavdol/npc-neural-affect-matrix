use lazy_static::lazy_static;
use std::sync::Mutex;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::api::types::NpcId;

lazy_static! {
    static ref NPC_MEMORIES: Mutex<HashMap<NpcId, Vec<MemoryRecord>>> = Mutex::new(HashMap::new());
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub source_id: String,
    pub text: String,
    pub valence: f32,
    pub arousal: f32,
    pub past_time: i64,
}

pub struct MemoryStore;

impl MemoryStore {
    pub fn new() -> Self {
        MemoryStore
    }

    pub fn insert(npc_id: &NpcId, record: MemoryRecord) -> Result<(), String> {
        let mut npc_memories = NPC_MEMORIES.lock().map_err(|_| "Failed to acquire lock")?;
        let npc_memory = npc_memories.entry(npc_id.clone()).or_insert_with(Vec::new);
        npc_memory.push(record);
        Ok(())
    }

    pub fn get_all(npc_id: &NpcId) -> Result<Vec<MemoryRecord>, String> {
        let npc_memories = NPC_MEMORIES.lock().map_err(|_| "Failed to acquire lock")?;
        Ok(npc_memories.get(npc_id).cloned().unwrap_or_default())
    }

    pub fn get_by_source(npc_id: &NpcId, source_id: &str) -> Result<Vec<MemoryRecord>, String> {
        let npc_memories = NPC_MEMORIES.lock().map_err(|_| "Failed to acquire lock")?;
        let empty_vec = Vec::new();
        let npc_memory = npc_memories.get(npc_id).unwrap_or(&empty_vec);
        Ok(npc_memory.iter()
            .filter(|record| record.source_id == source_id)
            .cloned()
            .collect())
    }

    pub fn import(npc_id: &NpcId, records: Vec<MemoryRecord>) -> Result<(), String> {
        for (index, record) in records.iter().enumerate() {
            if record.id.is_empty() {
                return Err(format!("Record at index {} has empty ID", index));
            }
            if record.valence < -1.0 || record.valence > 1.0 {
                return Err(format!("Record {} has invalid valence: {} (must be between -1.0 and 1.0)", record.id, record.valence));
            }
            if record.arousal < -1.0 || record.arousal > 1.0 {
                return Err(format!("Record {} has invalid arousal: {} (must be between -1.0 and 1.0)", record.id, record.arousal));
            }
        }

        let mut npc_memories = NPC_MEMORIES.lock().map_err(|_| "Failed to acquire lock")?;
        let npc_memory = npc_memories.entry(npc_id.clone()).or_insert_with(Vec::new);
        npc_memory.clear();
        npc_memory.extend(records);
        Ok(())
    }

    pub fn clear(npc_id: &NpcId) -> Result<(), String> {
        let mut npc_memories = NPC_MEMORIES.lock().map_err(|_| "Failed to acquire lock")?;
        if let Some(npc_memory) = npc_memories.get_mut(npc_id) {
            npc_memory.clear();
        }
        Ok(())
    }

    pub fn remove_npc(npc_id: &NpcId) -> Result<(), String> {
        let mut npc_memories = NPC_MEMORIES.lock().map_err(|_| "Failed to acquire lock")?;
        npc_memories.remove(npc_id);
        Ok(())
    }

    pub fn get_memory_count(npc_id: &NpcId) -> Result<usize, String> {
        let npc_memories = NPC_MEMORIES.lock().map_err(|_| "Failed to acquire lock")?;
        Ok(npc_memories.get(npc_id).map(|mem| mem.len()).unwrap_or(0))
    }

}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}


