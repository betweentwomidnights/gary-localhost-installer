//! Published Stable Audio Open Small model components. Each generation tier
//! is its complete base bundle (the ARC DiT, text encoder and decoder at that
//! tier); a finetune at a tier is just its DiT, run with that tier's encoder
//! and decoder, so it is an option of the tier rather than a part of it.

use crate::native_models::Catalog;
use std::sync::OnceLock;

pub const TIERS: &[&str] = &["F16", "Q8_0", "Q5_K_M", "Q4_K_M"];
/// The published finetunes, by the variant name sat-server and the catalog use.
pub const FINETUNES: &[&str] = &["kickbass", "jerry-grunge"];

pub fn models() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| Catalog::bundled("stable-audio", include_str!("../stable-audio-models.json")))
}

fn check_tier(tier: &str) -> Result<(), String> {
    if TIERS.contains(&tier) {
        Ok(())
    } else {
        Err("Choose a jerry encoding: F16, Q8_0, Q5_K_M or Q4_K_M.".into())
    }
}

/// What generating at `tier` needs: its base bundle.
pub fn tier_components(tier: &str) -> Result<Vec<String>, String> {
    check_tier(tier)?;
    Ok(vec![format!("stable-audio-native::{tier}")])
}

/// The finetunes that can run at `tier`, each in jerry's model dropdown once
/// it is downloaded.
pub fn finetunes(tier: &str) -> Vec<String> {
    if check_tier(tier).is_err() {
        return Vec::new();
    }
    FINETUNES
        .iter()
        .map(|name| format!("stable-audio-native::{name}::{tier}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tier_is_a_complete_pinned_bundle_with_its_finetunes() {
        let mut ids = std::collections::HashSet::new();
        for entry in models().components() {
            assert!(entry.id.starts_with("stable-audio-native::"));
            assert!(ids.insert(entry.id.as_str()), "{} is listed twice", entry.id);
            for file in &entry.files {
                file.validate().unwrap();
                assert_eq!(file.repo, "thepatch/stable-audio-open-small-GGUF");
            }
        }
        let names = |id: &str| -> Vec<String> {
            models().component(id).unwrap().files.iter().map(|file| file.filename.clone()).collect()
        };
        for tier in TIERS {
            // sat-server --encoding <tier> loads the DiT, text encoder and decoder at that tier,
            // and finds finetunes under finetunes/<variant>/ beside them.
            assert_eq!(names(&tier_components(tier).unwrap()[0]), [
                format!("stable-audio-open-small-dit-0.3B-v1.0-{tier}.gguf"),
                format!("t5-base-encoder-0.1B-v1.0-{tier}.gguf"),
                format!("stable-audio-open-small-oobleck-v1.0-{tier}.gguf"),
            ]);
            let options = finetunes(tier);
            assert_eq!(names(&options[0]), [format!("finetunes/kickbass/kickbass-v1-e257-dit-0.3B-v1.0-{tier}.gguf")]);
            assert_eq!(names(&options[1]), [
                format!("finetunes/jerry-grunge/jerry-grunge-bs128-step320-dit-0.3B-v1.0-{tier}.gguf"),
                "finetunes/jerry-grunge/prompts.json".to_string(),
            ]);
        }
        assert_eq!(models().components().len(), TIERS.len() * (1 + FINETUNES.len()));
        assert!(tier_components("../F16").is_err());
        assert!(finetunes("../F16").is_empty());
    }

    /// Removing one tier's Jerry Grunge keeps the prompt bank while another
    /// tier of it still needs it, and removes it with the last one.
    #[test]
    fn a_shared_prompt_bank_goes_with_the_last_finetune_that_needs_it() {
        let dir = std::env::temp_dir().join(format!("gary-saos-models-{}", std::process::id()));
        let place = |id: &str| {
            for file in &models().component(id).unwrap().files {
                let path = dir.join(&file.filename);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::File::create(path).unwrap().set_len(file.bytes).unwrap();
            }
        };
        let (q4, f16) = ("stable-audio-native::jerry-grunge::Q4_K_M", "stable-audio-native::jerry-grunge::F16");
        place(q4);
        place(f16);
        let removable = |id: &str| -> Vec<String> {
            models().removable_files(id, &dir).into_iter().map(|file| file.filename).collect()
        };
        assert_eq!(removable(q4), ["finetunes/jerry-grunge/jerry-grunge-bs128-step320-dit-0.3B-v1.0-Q4_K_M.gguf"]);
        std::fs::remove_file(dir.join("finetunes/jerry-grunge/jerry-grunge-bs128-step320-dit-0.3B-v1.0-F16.gguf")).unwrap();
        assert!(removable(q4).contains(&"finetunes/jerry-grunge/prompts.json".to_string()));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
