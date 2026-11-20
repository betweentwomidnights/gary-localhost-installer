//! SA3 dice prompt pools shared by desktop and headless training.
use regex::Regex;
use serde::Serialize;
use serde_json::json;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

fn prompt_from_caption(text: &str) -> String {
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

/// Build a LoRA's dice pool in an existing managed prompts directory.
pub fn build(dir: &Path, name: &str, dataset: &Path, force: bool) -> Result<String, String> {
    if crate::training::sanitize_lora_name(name).as_deref() != Some(name) {
        return Err("Invalid prompt pool name".into());
    }
    let dir = dir.canonicalize().map_err(|error| error.to_string())?;
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
    save_with_policy(
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

/// Publish complete JSON and preserve an existing file when overwrite is disabled.
pub fn save_with_policy(
    path: &Path,
    value: &impl Serialize,
    overwrite: bool,
) -> Result<(), String> {
    use std::io::Write;
    static NEXT_WRITE: AtomicU64 = AtomicU64::new(0);
    let stage = path.with_extension(format!(
        "json-{}-{}.tmp",
        std::process::id(),
        NEXT_WRITE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stage)
        .map_err(|error| error.to_string())?;
    let result = (|| {
        file.write_all(&serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        drop(file);
        if overwrite {
            std::fs::rename(&stage, path).map_err(|error| error.to_string())
        } else {
            match publish_new(&stage, path) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
                Err(error) => Err(error.to_string()),
            }
        }
    })();
    if stage.exists() {
        let _ = std::fs::remove_file(stage);
    }
    result
}

#[cfg(windows)]
fn publish_new(stage: &Path, dest: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{MoveFileExW, MOVEFILE_WRITE_THROUGH};
    let stage: Vec<u16> = stage.as_os_str().encode_wide().chain(Some(0)).collect();
    let dest: Vec<u16> = dest.as_os_str().encode_wide().chain(Some(0)).collect();
    // No REPLACE_EXISTING: preserve a curated pool created during caption scanning.
    // A rename also works on custom FAT/exFAT storage without hard-link support.
    if unsafe { MoveFileExW(stage.as_ptr(), dest.as_ptr(), MOVEFILE_WRITE_THROUGH) } != 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(windows))]
fn publish_new(stage: &Path, dest: &Path) -> std::io::Result<()> {
    std::fs::hard_link(stage, dest)
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
}
