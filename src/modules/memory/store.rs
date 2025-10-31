use crate::api::types::NpcId;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

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

        Ok(npc_memory
            .iter()
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
                return Err(format!(
                    "Record {} has invalid valence: {} (must be between -1.0 and 1.0)",
                    record.id, record.valence
                ));
            }
            if record.arousal < -1.0 || record.arousal > 1.0 {
                return Err(format!(
                    "Record {} has invalid arousal: {} (must be between -1.0 and 1.0)",
                    record.id, record.arousal
                ));
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

#[cfg(test)]
mod tests {
    use super::{MemoryRecord, MemoryStore};
    use crate::_test_mock::memory_mock::MemoryTestHelpers;

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
        let npc_id = MemoryTestHelpers::test_npc_id();
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
        let npc_id = MemoryTestHelpers::test_npc_id();
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
        let npc_id = MemoryTestHelpers::test_npc_id();
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
        let npc_id = MemoryTestHelpers::test_npc_id();
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
        let npc_id = MemoryTestHelpers::test_npc_id();
        MemoryStore::clear(&npc_id).unwrap();

        let records = vec![MemoryRecord {
            id: "".to_string(),
            source_id: "source-1".to_string(),
            text: "Invalid message".to_string(),
            valence: 0.5,
            arousal: -0.3,
            past_time: 1000,
        }];

        let result = MemoryStore::import(&npc_id, records);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty ID"));
    }

    #[test]
    fn test_memory_store_import_invalid_valence() {
        let npc_id = MemoryTestHelpers::test_npc_id();
        MemoryStore::clear(&npc_id).unwrap();

        let records = vec![MemoryRecord {
            id: "test-1".to_string(),
            source_id: "source-1".to_string(),
            text: "Invalid valence".to_string(),
            valence: 1.5,
            arousal: -0.3,
            past_time: 1000,
        }];

        let result = MemoryStore::import(&npc_id, records);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid valence"));
    }

    #[test]
    fn test_memory_store_import_invalid_arousal() {
        let npc_id = MemoryTestHelpers::test_npc_id();
        MemoryStore::clear(&npc_id).unwrap();

        let records = vec![MemoryRecord {
            id: "test-1".to_string(),
            source_id: "source-1".to_string(),
            text: "Invalid arousal".to_string(),
            valence: 0.5,
            arousal: -1.5,
            past_time: 1000,
        }];

        let result = MemoryStore::import(&npc_id, records);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid arousal"));
    }

    #[test]
    fn test_memory_store_clear() {
        let npc_id = MemoryTestHelpers::test_npc_id();
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
        let npc_id = MemoryTestHelpers::test_npc_id();
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
        let npc_id = MemoryTestHelpers::test_npc_id();
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
        let npc_id = MemoryTestHelpers::test_npc_id();
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
}
