use npc_neural_affect_matrix::config::{NpcConfig, Identity, PersonalityTraits, MemoryConfig};

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
        personality: PersonalityTraits { valence: 0.3, arousal: -0.2 },
        memory: MemoryConfig::new(0.15),
    };
    assert_eq!(custom.identity.name, "Test");
    assert_eq!(custom.personality.valence, 0.3);
    assert_eq!(custom.memory.decay_rate, 0.15);
}
