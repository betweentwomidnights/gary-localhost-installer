use crate::download::is_sha256;

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
            || ![".gguf", ".safetensors", ".json"]
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
    static CATALOG: std::sync::OnceLock<Vec<CatalogComponent>> = std::sync::OnceLock::new();
    let catalog = CATALOG.get_or_init(|| {
        serde_json::from_str(include_str!("../../src-tauri/sa3-models.json"))
            .expect("bundled SA3 catalog")
    });
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
