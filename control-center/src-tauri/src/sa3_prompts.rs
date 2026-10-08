//! Caption dice pools must remain available after the Python environment retires.
use regex::Regex;
use serde_json::json;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub fn prompt_from_caption(text: &str) -> String {
    static LABELED: OnceLock<Regex> = OnceLock::new();
    static BARE: OnceLock<Regex> = OnceLock::new();
    let labeled = LABELED.get_or_init(|| Regex::new(r"(?i)[,;]?\s*(?:bpm\s*[:=]?\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*bpm|(?:key|scale)\s*[:=]\s*[A-G][#b♯♭]?\s+(?:maj(?:or)?|min(?:or)?))\s*$").unwrap());
    let bare = BARE.get_or_init(|| Regex::new(r"(?i)[A-G][#b♯♭]?\s+(?:major|minor)\s*$").unwrap());
    let mut prompt = text.trim().to_string();
    loop {
        let start = labeled
            .find(&prompt)
            .map(|matched| matched.start())
            .or_else(|| {
                let matched = bare.find(&prompt)?;
                // Rust regex has no lookbehind. Preserve the legacy ASCII word boundary.
                (!prompt[..matched.start()]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_ascii_alphabetic()))
                .then_some(matched.start())
            });
        let stripped = prompt[..start.unwrap_or(prompt.len())]
            .trim_matches([' ', ',', ';', '\t', '\r', '\n'])
            .to_string();
        if stripped == prompt {
            return prompt;
        }
        prompt = stripped;
    }
}

fn captions(
    dir: &Path,
    boundary: &Path,
    files: &mut Vec<PathBuf>,
    visited: &mut BTreeSet<PathBuf>,
) -> Result<(), String> {
    if !visited.insert(dir.to_path_buf()) {
        return Ok(());
    }
    for file in std::fs::read_dir(dir).map_err(|error| error.to_string())? {
        let file = file.map_err(|error| error.to_string())?;
        // Do not follow directory links into a different dataset, or recurse in cycles.
        if file
            .file_type()
            .map_err(|error| error.to_string())?
            .is_symlink()
        {
            continue;
        }
        let path = file
            .path()
            .canonicalize()
            .map_err(|error| error.to_string())?;
        if !path.starts_with(boundary) {
            return Err("Caption path escapes its dataset folder".into());
        }
        if path.is_dir() {
            captions(&path, boundary, files, visited)?;
        } else if path
            .extension()
            .is_some_and(|ext| ext.to_string_lossy().eq_ignore_ascii_case("txt"))
        {
            files.push(path);
        }
    }
    Ok(())
}

pub fn build(root: &Path, name: &str, dataset: &Path, force: bool) -> Result<String, String> {
    if crate::sanitize_lora_name(name).as_deref() != Some(name) {
        return Err("Invalid prompt pool name".into());
    }
    let dir = crate::sa3_training::checked_folder(root, &["sa3", "prompts"])?;
    let dest = dir.join(format!("{name}.json"));
    if dest.exists() {
        if dest
            .canonicalize()
            .map_err(|error| error.to_string())?
            .parent()
            != Some(dir.as_path())
        {
            return Err("Prompt pool points outside managed storage".into());
        }
        if !force {
            return Ok(format!("Preserved existing prompts for {name}."));
        }
    }
    let boundary = dataset.canonicalize().map_err(|error| error.to_string())?;
    let mut files = Vec::new();
    captions(&boundary, &boundary, &mut files, &mut BTreeSet::new())?;
    files.sort_by_key(|path| {
        path.strip_prefix(&boundary)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/")
            .to_lowercase()
    });
    let mut seen = BTreeSet::new();
    let mut prompts = Vec::new();
    for path in &files {
        let data = std::fs::read(path).map_err(|error| error.to_string())?;
        let text = String::from_utf8_lossy(&data);
        let prompt = prompt_from_caption(text.trim_start_matches('\u{feff}'));
        if !prompt.is_empty() && seen.insert(prompt.to_lowercase()) {
            prompts.push(prompt);
        }
    }
    crate::sa3_training::save_with_policy(
        &dest,
        &json!({"version":1,"source":{"lora":name,"captions_dir":dataset,"files":files.len(),"unique_prompts":prompts.len()},"dice":{"instrumental":prompts}}),
        force,
    )?;
    Ok(format!(
        "Wrote prompts for {name}: {} unique prompts from {} captions.",
        seen.len(),
        files.len()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_legacy_caption_cleanup() {
        for (input, expected) in [
            ("warm analog synthwave", "warm analog synthwave"),
            ("beat, BPM: 92", "beat"),
            ("beat, 92 bpm", "beat"),
            ("pad, C minor", "pad"),
            ("pad, Key: F# major", "pad"),
            ("dnb, 174 bpm, A minor", "dnb"),
            ("dnb, A minor, 174 bpm", "dnb"),
            ("groove, 96 bpm, Bb major", "groove"),
            ("riff, Key: A min", "riff"),
            ("pad, E♭ minor", "pad"),
            ("pad, G♯ major", "pad"),
            ("dark booming minor", "dark booming minor"),
            (
                "C major scale run over swung drums",
                "C major scale run over swung drums",
            ),
            ("warm C min", "warm C min"),
        ] {
            assert_eq!(prompt_from_caption(input), expected, "{input}");
        }
    }
    #[test]
    fn pools_deduplicate_and_preserve_curated_prompts() {
        let root = std::env::temp_dir().join(format!("gary-native-prompts-{}", std::process::id()));
        std::fs::create_dir_all(root.join("dataset/sub")).unwrap();
        std::fs::write(root.join("dataset/a.txt"), "\u{feff}Dnb, 174 bpm, A minor").unwrap();
        std::fs::write(root.join("dataset/sub/b.txt"), "dnb, Key: C min").unwrap();
        build(&root, "test", &root.join("dataset"), false).unwrap();
        let path = root.join("sa3/prompts/test.json");
        let pool: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(pool["dice"]["instrumental"], json!(["Dnb"]));
        std::fs::write(&path, b"curated").unwrap();
        crate::sa3_training::save_with_policy(&path, &json!({"replacement":true}), false).unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"curated",
            "atomic publication must not replace a concurrently created pool"
        );
        build(&root, "test", &root.join("dataset"), false).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"curated");
        assert!(build(&root, "../escape", &root.join("dataset"), true).is_err());
        crate::remove_managed_path(&root, &std::env::temp_dir().canonicalize().unwrap()).unwrap();
    }
}
