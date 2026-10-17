//! Published Foundation-1 model components. Each generation tier is its
//! complete bundle (DiT, text encoder and decoder at that tier), and every tier
//! also gets the F16 encoder that audio2audio starts from.

use crate::native_models::Catalog;
use std::sync::OnceLock;

pub const ENCODER: &str = "foundation-native::encoder";
pub const TIERS: &[&str] = &["F16", "Q8_0", "Q5_K_M", "Q4_K_M"];

pub fn models() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| Catalog::bundled("foundation", include_str!("../foundation-models.json")))
}

/// What generating at `tier` needs: its bundle and the shared encoder, since
/// gary4juce's audio2audio is part of the service.
pub fn tier_components(tier: &str) -> Result<Vec<String>, String> {
    if !TIERS.contains(&tier) {
        return Err("Choose a Foundation-1 encoding: F16, Q8_0, Q5_K_M or Q4_K_M.".into());
    }
    Ok(vec![format!("foundation-native::{tier}"), ENCODER.into()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_tier_is_a_complete_pinned_bundle_with_the_shared_encoder() {
        let mut files = HashSet::new();
        for entry in models().components() {
            assert!(entry.id.starts_with("foundation-native::"));
            for file in &entry.files {
                file.validate().unwrap();
                assert_eq!(file.repo, "thepatch/foundation-1-GGUF");
                assert!(files.insert(&file.filename), "shared files belong in one component");
            }
        }
        for tier in TIERS {
            let ids = tier_components(tier).unwrap();
            assert_eq!(ids.last().unwrap(), ENCODER);
            let names: Vec<_> = models().component(&ids[0]).unwrap().files.iter().map(|file| file.filename.as_str()).collect();
            // sat-server --encoding <tier> loads the DiT, text encoder and decoder at that tier.
            assert_eq!(names, [
                format!("foundation-1-dit-1.1B-v1.0-{tier}.gguf"),
                format!("t5-base-encoder-128tok-0.1B-v1.0-{tier}.gguf"),
                format!("stable-audio-open-oobleck-v1.0-{tier}.gguf"),
            ]);
        }
        assert_eq!(
            models().component(ENCODER).unwrap().files[0].filename,
            "stable-audio-open-oobleck-encoder-v1.0-F16.gguf"
        );
        assert!(tier_components("../F16").is_err());
        assert!(models().component("foundation-native::../x").is_none());
    }
}
