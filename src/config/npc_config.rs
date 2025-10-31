use serde::{Deserialize, Serialize};
use std::convert::Into;

pub type PersonalityValue = f32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    pub name: String,
    pub background: String,
}

impl Identity {
    pub fn new(name: impl Into<String>, background: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            background: background.into(),
        }
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self::new("", "")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityTraits {
    pub valence: PersonalityValue,
    pub arousal: PersonalityValue,
}

impl PersonalityTraits {
    pub fn new() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.0,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let traits = [("valence", self.valence), ("arousal", self.arousal)];

        for (name, value) in traits {
            if value < -1.0 || value > 1.0 {
                return Err(format!(
                    "Personality trait '{}' has value {}, but must be between -1.0 and 1.0",
                    name, value
                ));
            }
        }

        Ok(())
    }
}

impl Default for PersonalityTraits {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub decay_rate: f32,
}

impl MemoryConfig {
    pub fn new(decay_rate: impl Into<f32>) -> Self {
        Self {
            decay_rate: decay_rate.into(),
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self { decay_rate: 0.1 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpcConfig {
    pub identity: Identity,
    pub personality: PersonalityTraits,
    pub memory: MemoryConfig,
}

impl Default for NpcConfig {
    fn default() -> Self {
        Self {
            identity: Identity::default(),
            personality: PersonalityTraits::default(),
            memory: MemoryConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Identity, MemoryConfig, NpcConfig, PersonalityTraits};

    #[test]
    fn test_npc_config_default() {
        let config = NpcConfig::default();
        assert_eq!(config.identity.name, "");
        assert_eq!(config.identity.background, "");
        assert_eq!(config.personality.valence, 0.0);
        assert_eq!(config.personality.arousal, 0.0);
        assert_eq!(config.memory.decay_rate, 0.1);
    }

    #[test]
    fn test_identity() {
        let identity = Identity::new("Alice", "Warrior");
        assert_eq!(identity.name, "Alice");
        assert_eq!(identity.background, "Warrior");

        let default = Identity::default();
        assert_eq!(default.name, "");
        assert_eq!(default.background, "");
    }

    #[test]
    fn test_personality_traits() {
        let mut traits = PersonalityTraits::new();
        assert_eq!(traits.valence, 0.0);
        assert_eq!(traits.arousal, 0.0);

        traits.valence = 0.5;
        traits.arousal = -0.3;
        assert!(traits.validate().is_ok());

        traits.valence = 1.5;
        assert!(traits.validate().is_err());
    }

    #[test]
    fn test_memory_config() {
        let memory = MemoryConfig::new(0.5);
        assert_eq!(memory.decay_rate, 0.5);

        let default = MemoryConfig::default();
        assert_eq!(default.decay_rate, 0.1);
    }

    #[test]
    fn test_npc_config() {
        let config = NpcConfig::default();
        assert_eq!(config.identity.name, "");
        assert_eq!(config.personality.valence, 0.0);
        assert_eq!(config.memory.decay_rate, 0.1);

        let custom = NpcConfig {
            identity: Identity::new("Test", "Character"),
            personality: PersonalityTraits {
                valence: 0.3,
                arousal: -0.2,
            },
            memory: MemoryConfig::new(0.15),
        };
        assert_eq!(custom.identity.name, "Test");
        assert_eq!(custom.personality.valence, 0.3);
        assert_eq!(custom.memory.decay_rate, 0.15);
    }
}
