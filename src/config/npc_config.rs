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
        Self { name: name.into(), background: background.into() }
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
        let traits = [
            ("valence", self.valence),
            ("arousal", self.arousal),
        ];

        for (name, value) in traits {
            if value < -1.0 || value > 1.0 {
                return Err(format!("Personality trait '{}' has value {}, but must be between -1.0 and 1.0", name, value));
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
        Self { decay_rate: decay_rate.into() }
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
