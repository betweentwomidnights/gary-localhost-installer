//! Published ACE-Step 1.5 model components for native Carey. ace-server keeps
//! one registry of every GGUF in its models folder and each request names the
//! file it wants, so a tier is only what gets downloaded: the 2B base DiT at
//! that tier with the text encoder and VAE every model shares. The 2B family's
//! sft and turbo and the LMs are options of a tier. The XL family fits a GPU
//! differently, so it is offered at its own recommended tier instead.

use crate::native_models::Catalog;
use std::sync::OnceLock;

pub const TIERS: &[&str] = &["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"];
/// The DiT a tier needs: every task (lego, extract and complete included)
/// runs on it, and it is the model Python Carey defaults to.
pub const DEFAULT_DIT: &str = "base";
/// The other 2B DiTs, offered at a tier alongside its base, by their GGUF
/// names' variant.
pub const SIBLING_DITS: &[&str] = &["sft", "turbo"];
/// The XL DiTs, which are recommended at their own tier.
pub const XL_DITS: &[&str] = &["xl-base", "xl-sft", "xl-turbo"];
pub const SHARED: &str = "carey-native::shared";

/// GPU memory kept free beside a DiT's weights: its activations, the text
/// encoder, the backend's context and room for a DAW. ace-server unloads the
/// DiT before decoding and sizes VAE tiles to what is then free, so whether a
/// tier generates comes down to its DiT's weights. On an 8 GB RTX 5070 this
/// recommends the 2B base at BF16 and XL at Q8_0, both of which rendered 120 s
/// there, and rules out XL BF16, whose weights alone exceed the card.
pub const HEADROOM_BYTES: u64 = 2 << 30;

pub fn models() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| Catalog::bundled("carey", include_str!("../carey-models.json")))
}

fn check_tier(tier: &str) -> Result<(), String> {
    if TIERS.contains(&tier) {
        Ok(())
    } else {
        Err("Choose a Carey encoding: BF16, Q8_0, Q6_K, Q5_K_M or Q4_K_M.".into())
    }
}

pub fn dit_component(variant: &str, tier: &str) -> String {
    format!("carey-native::{variant}::{tier}")
}

/// The GGUF name ace-server knows a DiT by, which is what a request's
/// `synth_model` names.
pub fn dit_file(variant: &str, tier: &str) -> String {
    format!("acestep-v15-{variant}-{tier}.gguf")
}

/// The LM builds that go with `tier`: the small LMs come in BF16 and Q8_0,
/// the 4B down to Q5_K_M, as acestep.cpp's models.sh resolves them.
fn lm_components(tier: &str) -> Vec<String> {
    let small = if tier == "BF16" { "BF16" } else { "Q8_0" };
    let large = match tier {
        "Q5_K_M" | "Q4_K_M" => "Q5_K_M",
        other => other,
    };
    vec![
        format!("carey-native::lm-0.6B::{small}"),
        format!("carey-native::lm-1.7B::{small}"),
        format!("carey-native::lm-4B::{large}"),
    ]
}

/// What generating at `tier` needs: its base DiT, then what every DiT shares.
pub fn tier_components(tier: &str) -> Result<Vec<String>, String> {
    check_tier(tier)?;
    Ok(vec![dit_component(DEFAULT_DIT, tier), SHARED.to_string()])
}

/// The other 2B DiTs at `tier` and the LMs that go with it.
pub fn options(tier: &str) -> Vec<String> {
    if check_tier(tier).is_err() {
        return Vec::new();
    }
    SIBLING_DITS
        .iter()
        .map(|variant| dit_component(variant, tier))
        .chain(lm_components(tier))
        .collect()
}

/// The best tier of `variant` whose DiT fits a GPU with `memory_bytes` beside
/// the headroom; the smallest tier when none does.
pub fn recommended_tier(variant: &str, memory_bytes: u64) -> &'static str {
    TIERS
        .iter()
        .copied()
        .find(|tier| {
            models().component(&dit_component(variant, tier)).is_some_and(|component| {
                component.files.iter().map(|file| file.bytes).sum::<u64>() + HEADROOM_BYTES <= memory_bytes
            })
        })
        .unwrap_or(TIERS[TIERS.len() - 1])
}

/// The tier to prepare (the base DiT's), and every DiT at its own best tier:
/// the two families fit differently, so each gets its own recommendation.
pub fn recommendations(memory_bytes: u64) -> crate::migration::Recommendations {
    crate::migration::Recommendations {
        tier: Some(recommended_tier(DEFAULT_DIT, memory_bytes).to_string()),
        components: std::iter::once(DEFAULT_DIT)
            .chain(SIBLING_DITS.iter().copied())
            .chain(XL_DITS.iter().copied())
            .map(|variant| dit_component(variant, recommended_tier(variant, memory_bytes)))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn each_family_gets_the_largest_tier_its_weights_fit() {
        // The RTX 5070 Laptop GPU these were measured on reports 8151 MiB.
        let laptop = recommendations(8151 * MIB);
        assert_eq!(laptop.tier.as_deref(), Some("BF16"));
        assert_eq!(laptop.components, [
            "carey-native::base::BF16",
            "carey-native::sft::BF16",
            "carey-native::turbo::BF16",
            "carey-native::xl-base::Q8_0",
            "carey-native::xl-sft::Q8_0",
            "carey-native::xl-turbo::Q8_0",
        ]);
        assert_eq!(recommended_tier("xl-base", 16 * 1024 * MIB), "BF16");
        assert_eq!(recommended_tier("base", 6 * 1024 * MIB), "Q8_0");
        assert_eq!(recommended_tier("xl-base", 6 * 1024 * MIB), "Q6_K");
        assert_eq!(recommended_tier("xl-base", 2 * 1024 * MIB), "Q4_K_M");
    }

    #[test]
    fn every_component_is_pinned_and_every_tier_resolves_to_published_files() {
        let mut ids = std::collections::HashSet::new();
        for entry in models().components() {
            assert!(entry.id.starts_with("carey-native::"));
            assert!(ids.insert(entry.id.as_str()), "{} is listed twice", entry.id);
            for file in &entry.files {
                file.validate().unwrap();
                assert_eq!(file.repo, "Serveurperso/ACE-Step-1.5-GGUF");
            }
        }
        let names = |id: &str| -> Vec<String> {
            models()
                .component(id)
                .unwrap_or_else(|| panic!("{id} is not in the catalog"))
                .files
                .iter()
                .map(|file| file.filename.clone())
                .collect()
        };
        assert_eq!(names(SHARED), ["Qwen3-Embedding-0.6B-Q8_0.gguf", "vae-BF16.gguf"]);
        let mut offered = std::collections::HashSet::new();
        for tier in TIERS {
            let needed = tier_components(tier).unwrap();
            assert_eq!(needed[1], SHARED);
            assert_eq!(names(&needed[0]), [dit_file(DEFAULT_DIT, tier)]);
            let options = options(tier);
            for (variant, id) in SIBLING_DITS.iter().zip(&options) {
                assert_eq!(names(id), [dit_file(variant, tier)]);
            }
            for variant in XL_DITS {
                let id = dit_component(variant, tier);
                assert_eq!(names(&id), [dit_file(variant, tier)]);
                offered.insert(id);
            }
            for id in &options[SIBLING_DITS.len()..] {
                let (size, quant) = id["carey-native::lm-".len()..].split_once("::").unwrap();
                assert_eq!(names(id), [format!("acestep-5Hz-lm-{size}-{quant}.gguf")]);
            }
            offered.insert(needed[0].clone());
            offered.extend(options);
        }
        // Everything but the shared component is a tier's, an option of one,
        // or an XL DiT at some tier, so the models panel shows no stray download.
        offered.insert(SHARED.to_string());
        assert_eq!(offered.len(), models().components().len());
        assert!(tier_components("../BF16").is_err());
        assert!(options("../BF16").is_empty());
    }

    #[test]
    fn quantized_tiers_take_the_lm_builds_models_sh_takes() {
        assert_eq!(&options("BF16")[2..], [
            "carey-native::lm-0.6B::BF16",
            "carey-native::lm-1.7B::BF16",
            "carey-native::lm-4B::BF16",
        ]);
        assert_eq!(&options("Q6_K")[2..], [
            "carey-native::lm-0.6B::Q8_0",
            "carey-native::lm-1.7B::Q8_0",
            "carey-native::lm-4B::Q6_K",
        ]);
        assert_eq!(options("Q4_K_M")[4], "carey-native::lm-4B::Q5_K_M");
    }
}
