use crate::download::is_sha256;
use crate::service::Service;

/// A model artifact pinned with the application, independently of HF's main.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PinnedHfFile {
    pub repo: String,
    pub revision: String,
    pub filename: String,
    pub sha256: String,
    pub bytes: u64,
}

impl PinnedHfFile {
    pub fn validate(&self) -> Result<(), String> {
        let parts: Vec<_> = self.repo.split('/').collect();
        let safe_part = |part: &str| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                && part != "."
                && part != ".."
        };
        // A file may sit in a repo subfolder (SAOS finetunes do), and keeps
        // that relative path under the models folder.
        if parts.len() != 2
            || !parts.iter().all(|part| safe_part(part))
            || self.revision.len() != 40
            || !self.revision.bytes().all(|b| b.is_ascii_hexdigit())
            || !self.filename.split('/').all(safe_part)
            || ![".gguf", ".safetensors", ".json", ".tiktoken"]
                .iter()
                .any(|ext| self.filename.ends_with(ext))
            || !is_sha256(&self.sha256)
            || self.bytes == 0
        {
            return Err("Invalid pinned Hugging Face model artifact.".into());
        }
        Ok(())
    }

    pub fn url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, self.filename
        )
    }
}

/// A downloadable part of a native service's models: pinned files that are
/// fetched, verified and kept together.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct CatalogComponent {
    pub id: String,
    pub label: String,
    pub description: String,
    pub files: Vec<PinnedHfFile>,
}

pub fn sa3_medium_files(
    model: &str,
    encoding: &str,
    t5: &str,
    ae: &str,
) -> Result<Vec<&'static PinnedHfFile>, String> {
    if !model.eq_ignore_ascii_case("medium")
        || !t5.eq_ignore_ascii_case("f16")
        || !ae.eq_ignore_ascii_case("f32")
    {
        return Err(
            "Managed downloads support SA3 medium with F16 text encoder and F32 decoder.".into(),
        );
    }
    let catalog = catalog(Service::Sa3);
    let tier = encoding.to_ascii_uppercase();
    if !["F16", "Q8_0", "Q5_K_M", "Q4_K_M"].contains(&tier.as_str()) {
        return Err(format!("Unsupported SA3 inference precision: {encoding}"));
    }
    let mut files = Vec::new();
    for id in [
        "sa3-native::text",
        "sa3-native::medium-decoder",
        &format!("sa3-native::medium-{tier}"),
    ] {
        let component = catalog
            .iter()
            .find(|item| item.id == id)
            .ok_or_else(|| format!("Missing catalog component {id}"))?;
        for file in &component.files {
            file.validate()?;
            files.push(file);
        }
    }
    Ok(files)
}

/// The variant's own catalog component. Shared encoder/decoder components
/// have independent identities and are kept when a variant is removed.
pub fn removable_variant_files(
    service: Service,
    model: &str,
    encoding: &str,
    dir: &std::path::Path,
) -> Result<Vec<&'static PinnedHfFile>, String> {
    let tier = encoding.to_ascii_uppercase();
    let id = match service {
        Service::Sa3 => format!("sa3-native::medium-{tier}"),
        Service::Gary | Service::Carey => format!("{}-native::{model}::{tier}", service.id()),
        Service::Yuey => tier,
        Service::StableAudio if model != "saos" && model != "arc" => {
            format!("stable-audio-native::{model}::{tier}")
        }
        _ => format!("{}-native::{tier}", service.id()),
    };
    let catalog = catalog(service);
    if !catalog.iter().any(|item| item.id == id) {
        return Err(format!("Unknown model component {id}"));
    }
    let files = removable_component_files(catalog, &id, dir);
    for file in &files {
        file.validate()?;
    }
    Ok(files)
}

pub fn removable_component_files<'a>(
    catalog: &'a [CatalogComponent],
    id: &str,
    dir: &std::path::Path,
) -> Vec<&'a PinnedHfFile> {
    let Some(component) = catalog.iter().find(|item| item.id == id) else {
        return Vec::new();
    };
    let mut files = Vec::new();
    for file in &component.files {
        if !catalog.iter().any(|other| {
            other.id != id
                && other
                    .files
                    .iter()
                    .any(|shared| shared.filename == file.filename)
                && other.files.iter().all(|required| {
                    std::fs::symlink_metadata(dir.join(&required.filename))
                        .is_ok_and(|meta| meta.is_file() && meta.len() == required.bytes)
                })
        }) {
            files.push(file);
        }
    }
    files
}

pub fn service_files(
    service: crate::service::Service,
    model: &str,
    encoding: &str,
) -> Result<Vec<&'static PinnedHfFile>, String> {
    use crate::service::Service::*;
    let tier = encoding.to_ascii_uppercase();
    let catalog = catalog(service);
    let ids = match service {
        Gary => vec![
            "gary-native::shared".into(),
            format!("gary-native::{model}::{tier}"),
        ],
        Melodyflow => vec![format!("melodyflow-native::{tier}")],
        StableAudio if model == "saos" || model == "arc" => {
            vec![format!("stable-audio-native::{tier}")]
        }
        StableAudio => vec![format!("stable-audio-native::{model}::{tier}")],
        Foundation => vec![
            format!("foundation-native::{tier}"),
            "foundation-native::encoder".into(),
        ],
        Carey => vec![
            "carey-native::shared".into(),
            format!("carey-native::{model}::{tier}"),
        ],
        Yuey => vec!["shared".into(), tier],
        Sa3 => return Err("Use the SA3 component encodings".into()),
    };
    let mut files = Vec::new();
    for id in ids {
        let component = catalog
            .iter()
            .find(|c| c.id == id)
            .ok_or_else(|| format!("Unknown model component {id}"))?;
        for file in &component.files {
            file.validate()?;
            files.push(file);
        }
    }
    Ok(files)
}

pub fn component(service: Service, id: &str) -> Option<&'static CatalogComponent> {
    catalog(service).iter().find(|item| item.id == id)
}

fn catalog(service: Service) -> &'static [CatalogComponent] {
    macro_rules! catalog {
        ($file:literal) => {{
            static ITEMS: std::sync::OnceLock<Vec<CatalogComponent>> = std::sync::OnceLock::new();
            ITEMS.get_or_init(|| {
                serde_json::from_str(include_str!($file)).expect("bundled model catalog")
            })
        }};
    }
    match service {
        Service::Sa3 => catalog!("../../src-tauri/sa3-models.json"),
        Service::Gary => catalog!("../../src-tauri/gary-models.json"),
        Service::Melodyflow => catalog!("../../src-tauri/melodyflow-models.json"),
        Service::StableAudio => catalog!("../../src-tauri/stable-audio-models.json"),
        Service::Foundation => catalog!("../../src-tauri/foundation-models.json"),
        Service::Carey => catalog!("../../src-tauri/carey-models.json"),
        Service::Yuey => catalog!("../yuey-models.json"),
    }
}

pub const PRECISIONS: &[&str] = &["BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"];

/// Inference choices in the published catalog. Training bases, LMs and shared
/// components are independent of a generation checkpoint's precision.
pub fn variants(service: Service) -> Vec<(&'static str, &'static str)> {
    let mut variants = Vec::new();
    for component in catalog(service) {
        let id = component.id.as_str();
        let pair = match service {
            Service::Sa3 => id
                .strip_prefix("sa3-native::medium-")
                .map(|tier| ("medium", tier)),
            Service::Yuey => Some(("yue2", id)),
            Service::Gary | Service::Carey => id
                .split_once("::")
                .and_then(|(_, id)| id.split_once("::"))
                .filter(|(model, _)| !model.starts_with("lm-")),
            _ => id
                .split_once("::")
                .map(|(_, id)| id.split_once("::").unwrap_or((service.defaults().0, id))),
        };
        if let Some((model, tier)) = pair.filter(|(_, tier)| PRECISIONS.contains(tier)) {
            variants.push((model, tier));
        }
    }
    variants.sort_by_key(|(model, tier)| {
        (
            (*model != service.defaults().0),
            *model,
            PRECISIONS.iter().position(|t| t == tier),
        )
    });
    variants
}

pub const GARY_HEADROOM_BYTES: u64 = 3 << 30;
pub const CAREY_HEADROOM_BYTES: u64 = 2 << 30;

/// Shared with the desktop recommendations. Budget the checkpoint's weights
/// plus headroom measured in the desktop generation workloads.
pub fn recommended_model_precision(
    service: Service,
    model: &str,
    memory_bytes: u64,
) -> &'static str {
    let headroom = match service {
        Service::Gary => GARY_HEADROOM_BYTES,
        Service::Carey => CAREY_HEADROOM_BYTES,
        _ => return "",
    };
    let choices: Vec<_> = variants(service)
        .into_iter()
        .filter(|(name, _)| *name == model)
        .collect();
    choices
        .iter()
        .find(|(name, tier)| {
            let id = format!("{}-native::{name}::{tier}", service.id());
            catalog(service)
                .iter()
                .find(|component| component.id == id)
                .is_some_and(|component| {
                    component.files.iter().map(|file| file.bytes).sum::<u64>() + headroom
                        <= memory_bytes
                })
        })
        .or_else(|| choices.last())
        .map_or("", |(_, tier)| *tier)
}

/// Every file this service publishes, for matching a file on disk by name.
pub fn all_files(service: Service) -> impl Iterator<Item = &'static PinnedHfFile> {
    catalog(service).iter().flat_map(|component| &component.files)
}

pub fn installed_carey_lms(
    dir: &std::path::Path,
) -> impl Iterator<Item = &'static PinnedHfFile> + '_ {
    catalog(Service::Carey)
        .iter()
        .filter(|component| component.id.starts_with("carey-native::lm-"))
        .flat_map(|component| &component.files)
        .filter(|file| {
            std::fs::metadata(dir.join(&file.filename))
                .is_ok_and(|stat| stat.is_file() && stat.len() == file.bytes)
        })
}

pub fn recommended_yuey_tier(encoding: &str) -> Option<&'static str> {
    match encoding {
        "BF16" => Some("BF16"),
        "Q8_0" => Some("Q8_0"),
        "Q5_K_M" | "Q4_K_M" => Some("Q4_K_M"),
        _ => None,
    }
}

#[cfg(test)]
mod recommendation_tests {
    use super::*;

    #[test]
    fn carey_precision_is_independent_for_2b_and_xl() {
        assert_eq!(
            recommended_model_precision(Service::Carey, "base", 8 << 30),
            "BF16"
        );
        assert_eq!(
            recommended_model_precision(Service::Carey, "xl-base", 8 << 30),
            "Q8_0"
        );
        assert_eq!(
            recommended_model_precision(Service::Carey, "xl-base", 91 << 30),
            "BF16"
        );
    }
}
