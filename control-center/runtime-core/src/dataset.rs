//! Sidecar inventory and format shared by desktop and the terminal.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CareyAceSidecarFields {
    #[serde(default)]
    pub caption: String,
    #[serde(default)]
    pub genre: String,
    #[serde(default)]
    pub bpm: String,
    #[serde(default)]
    pub bpm_source: String,
    #[serde(default)]
    pub lm_bpm: String,
    #[serde(default)]
    pub local_bpm: String,
    #[serde(default)]
    pub filename_bpm: String,
    #[serde(default)]
    pub keyscale: String,
    #[serde(default)]
    pub key_source: String,
    #[serde(default)]
    pub lm_keyscale: String,
    #[serde(default)]
    pub local_keyscale: String,
    #[serde(default)]
    pub timesignature: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub is_instrumental: bool,
    #[serde(default)]
    pub custom_tag: String,
    #[serde(default)]
    pub lyrics: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CareyAceDatasetSidecarEntry {
    pub audio_path: String,
    pub relative_path: String,
    pub sidecar_path: String,
    pub exists: bool,
    pub raw_content: String,
    pub fields: CareyAceSidecarFields,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sa3DatasetSidecarEntry {
    pub audio_path: String,
    pub relative_path: String,
    pub sidecar_path: String,
    pub content: String,
    pub exists: bool,
    pub json_sidecar_exists: bool,
}

pub fn read_dir_sorted(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut entries = Vec::new();
    let reader =
        std::fs::read_dir(dir).map_err(|e| format!("Cannot read {}: {}", dir.display(), e))?;

    for entry in reader {
        let entry = entry.map_err(|e| format!("Cannot read entry in {}: {}", dir.display(), e))?;
        entries.push(entry.path());
    }

    entries.sort_by(|a, b| {
        a.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .cmp(&b.file_name().unwrap_or_default().to_string_lossy())
    });
    Ok(entries)
}

pub fn is_sa3_dataset_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "wav" | "flac" | "mp3" | "ogg" | "opus" | "m4a" | "aiff" | "aif"
            )
        })
        .unwrap_or(false)
}

pub fn user_facing_windows_path(path: &Path) -> String {
    let value = path.to_string_lossy();
    if let Some(unc_path) = value.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{}", unc_path);
    }
    value.strip_prefix(r"\\?\").unwrap_or(&value).to_string()
}

pub fn canonical_sa3_dataset_root(dataset_path: &str) -> Result<PathBuf, String> {
    let trimmed = dataset_path.trim();
    if trimmed.is_empty() {
        return Err("Choose a dataset folder first".to_string());
    }
    let requested = PathBuf::from(trimmed);
    if !requested.is_dir() {
        return Err(format!(
            "{} is not a valid dataset folder",
            requested.display()
        ));
    }
    requested
        .canonicalize()
        .map_err(|e| format!("Cannot resolve {}: {}", requested.display(), e))
}

pub fn collect_sa3_dataset_audio_files(
    root: &Path,
    directory: &Path,
    visited: &mut std::collections::HashSet<PathBuf>,
    files: &mut Vec<PathBuf>,
) -> Result<(), String> {
    collect_dataset_audio_files(root, directory, visited, files, is_sa3_dataset_audio_file)
}

fn collect_dataset_audio_files(
    root: &Path,
    directory: &Path,
    visited: &mut std::collections::HashSet<PathBuf>,
    files: &mut Vec<PathBuf>,
    is_audio: fn(&Path) -> bool,
) -> Result<(), String> {
    let canonical_directory = directory
        .canonicalize()
        .map_err(|e| format!("Cannot resolve {}: {}", directory.display(), e))?;
    if !canonical_directory.starts_with(root) || !visited.insert(canonical_directory.clone()) {
        return Ok(());
    }

    for entry in read_dir_sorted(&canonical_directory)? {
        let Ok(metadata) = std::fs::metadata(&entry) else {
            continue;
        };
        if metadata.is_dir() {
            collect_dataset_audio_files(root, &entry, visited, files, is_audio)?;
        } else if metadata.is_file() && is_audio(&entry) {
            // Pair linked audio with the sidecar beside the link, using its name.
            files.push(entry);
        }
    }
    Ok(())
}

/// Validate a track path from the dataset editor while retaining its local name.
pub fn resolve_dataset_audio_entry(
    root: &Path,
    requested: &Path,
    is_audio: fn(&Path) -> bool,
) -> Result<PathBuf, String> {
    if let (Some(parent), Some(name)) = (requested.parent(), requested.file_name()) {
        let directory = parent
            .canonicalize()
            .map_err(|e| format!("Cannot resolve {}: {}", parent.display(), e))?;
        let entry = directory.join(name);
        if directory.starts_with(root) && is_audio(&entry) && entry.is_file() {
            return Ok(entry);
        }
    }
    Err(format!(
        "{} is not an audio file inside {}",
        requested.display(),
        root.display()
    ))
}

/// Allow creating a sidecar or editing a regular file beside the track.
pub fn check_writable_sidecar(path: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(()),
        Ok(_) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Use a regular sidecar file at {}", path.display()),
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn sa3_dataset_json_sidecar_exists(root: &Path, audio_path: &Path) -> bool {
    let mut candidates = vec![audio_path.with_extension("json")];
    if let (Some(parent), Some(stem)) = (audio_path.parent(), audio_path.file_stem()) {
        if let Some(grandparent) = parent.parent() {
            candidates.push(grandparent.join("json").join(stem).with_extension("json"));
        }
    }
    candidates
        .into_iter()
        .any(|candidate| candidate.starts_with(root) && candidate.is_file())
}

pub fn read_sa3_dataset_sidecars(
    dataset_path: &str,
) -> Result<Vec<Sa3DatasetSidecarEntry>, String> {
    let root = canonical_sa3_dataset_root(dataset_path)?;
    let mut files = Vec::new();
    let mut visited = std::collections::HashSet::new();
    collect_sa3_dataset_audio_files(&root, &root, &mut visited, &mut files)?;
    files.sort_by(|left, right| {
        left.strip_prefix(&root)
            .unwrap_or(left)
            .to_string_lossy()
            .to_lowercase()
            .cmp(
                &right
                    .strip_prefix(&root)
                    .unwrap_or(right)
                    .to_string_lossy()
                    .to_lowercase(),
            )
    });

    Ok(files
        .into_iter()
        .map(|audio_path| {
            let sidecar_path = audio_path.with_extension("txt");
            let exists = sidecar_path.is_file();
            let content = if exists {
                std::fs::read_to_string(&sidecar_path)
                    .unwrap_or_default()
                    .trim_start_matches('\u{feff}')
                    .to_string()
            } else {
                String::new()
            };
            Sa3DatasetSidecarEntry {
                relative_path: audio_path
                    .strip_prefix(&root)
                    .unwrap_or(&audio_path)
                    .to_string_lossy()
                    .to_string(),
                audio_path: user_facing_windows_path(&audio_path),
                sidecar_path: user_facing_windows_path(&sidecar_path),
                content,
                exists,
                json_sidecar_exists: sa3_dataset_json_sidecar_exists(&root, &audio_path),
            }
        })
        .collect())
}

pub fn is_carey_ace_dataset_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "wav" | "flac" | "mp3" | "ogg" | "opus" | "m4a"
            )
        })
        .unwrap_or(false)
}

pub fn canonical_carey_ace_dataset_root(dataset_path: &str) -> Result<PathBuf, String> {
    let trimmed = dataset_path.trim();
    if trimmed.is_empty() {
        return Err("Choose a dataset folder first".to_string());
    }
    let requested = PathBuf::from(trimmed);
    if !requested.is_dir() {
        return Err(format!(
            "{} is not a valid dataset folder",
            requested.display()
        ));
    }
    requested
        .canonicalize()
        .map_err(|e| format!("Cannot resolve {}: {}", requested.display(), e))
}

fn normalize_carey_ace_sidecar_key(raw: &str) -> Option<&'static str> {
    match raw.trim().to_ascii_lowercase().replace(' ', "_").as_str() {
        "caption" => Some("caption"),
        "genre" => Some("genre"),
        "bpm" => Some("bpm"),
        "bpm_source" => Some("bpm_source"),
        "lm_bpm" => Some("lm_bpm"),
        "local_bpm" => Some("local_bpm"),
        "filename_bpm" => Some("filename_bpm"),
        "key" | "keyscale" => Some("keyscale"),
        "key_source" => Some("key_source"),
        "lm_keyscale" => Some("lm_keyscale"),
        "local_keyscale" => Some("local_keyscale"),
        "signature" | "timesignature" | "time_signature" => Some("timesignature"),
        "language" => Some("language"),
        "is_instrumental" => Some("is_instrumental"),
        "custom_tag" => Some("custom_tag"),
        "lyrics" => Some("lyrics"),
        _ => None,
    }
}

fn commit_carey_ace_sidecar_value(
    values: &mut BTreeMap<String, String>,
    key: &mut Option<&'static str>,
    lines: &mut Vec<String>,
) {
    if let Some(name) = key.take() {
        values.insert(name.to_string(), lines.join("\n").trim().to_string());
        lines.clear();
    }
}

pub fn parse_carey_ace_sidecar_content(raw: &str) -> CareyAceSidecarFields {
    let mut values = BTreeMap::<String, String>::new();
    let mut current_key: Option<&'static str> = None;
    let mut current_lines: Vec<String> = Vec::new();
    let mut saw_known_key = false;

    for line in raw.trim_start_matches('\u{feff}').lines() {
        let parsed = line.split_once(':').and_then(|(key, value)| {
            if key
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == ' ')
            {
                normalize_carey_ace_sidecar_key(key).map(|name| (name, value.trim_start()))
            } else {
                None
            }
        });

        if let Some((name, value)) = parsed {
            commit_carey_ace_sidecar_value(&mut values, &mut current_key, &mut current_lines);
            saw_known_key = true;
            current_key = Some(name);
            current_lines.push(value.trim_end().to_string());
        } else if current_key.is_some() {
            current_lines.push(line.trim_end().to_string());
        }
    }
    commit_carey_ace_sidecar_value(&mut values, &mut current_key, &mut current_lines);

    if !saw_known_key && !raw.trim().is_empty() {
        return CareyAceSidecarFields {
            lyrics: raw.trim().to_string(),
            ..CareyAceSidecarFields::default()
        };
    }

    CareyAceSidecarFields {
        caption: values.remove("caption").unwrap_or_default(),
        genre: values.remove("genre").unwrap_or_default(),
        bpm: values.remove("bpm").unwrap_or_default(),
        bpm_source: values.remove("bpm_source").unwrap_or_default(),
        lm_bpm: values.remove("lm_bpm").unwrap_or_default(),
        local_bpm: values.remove("local_bpm").unwrap_or_default(),
        filename_bpm: values.remove("filename_bpm").unwrap_or_default(),
        keyscale: values.remove("keyscale").unwrap_or_default(),
        key_source: values.remove("key_source").unwrap_or_default(),
        lm_keyscale: values.remove("lm_keyscale").unwrap_or_default(),
        local_keyscale: values.remove("local_keyscale").unwrap_or_default(),
        timesignature: values.remove("timesignature").unwrap_or_default(),
        language: values.remove("language").unwrap_or_default(),
        is_instrumental: values
            .remove("is_instrumental")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false),
        custom_tag: values.remove("custom_tag").unwrap_or_default(),
        lyrics: values.remove("lyrics").unwrap_or_default(),
    }
}

fn clean_carey_ace_scalar(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn render_carey_ace_sidecar(fields: &CareyAceSidecarFields) -> String {
    let mut lines = Vec::new();
    for (key, value) in [
        ("caption", fields.caption.as_str()),
        ("genre", fields.genre.as_str()),
        ("bpm", fields.bpm.as_str()),
        ("bpm_source", fields.bpm_source.as_str()),
        ("lm_bpm", fields.lm_bpm.as_str()),
        ("local_bpm", fields.local_bpm.as_str()),
        ("filename_bpm", fields.filename_bpm.as_str()),
        ("keyscale", fields.keyscale.as_str()),
        ("key_source", fields.key_source.as_str()),
        ("lm_keyscale", fields.lm_keyscale.as_str()),
        ("local_keyscale", fields.local_keyscale.as_str()),
        ("timesignature", fields.timesignature.as_str()),
        ("language", fields.language.as_str()),
    ] {
        let scalar = clean_carey_ace_scalar(value);
        if !scalar.is_empty() {
            lines.push(format!("{}: {}", key, scalar));
        }
    }
    lines.push(format!(
        "is_instrumental: {}",
        if fields.is_instrumental {
            "true"
        } else {
            "false"
        }
    ));
    let tag = clean_carey_ace_scalar(&fields.custom_tag);
    if !tag.is_empty() {
        lines.push(format!("custom_tag: {}", tag));
    }

    let mut lyrics = fields.lyrics.trim().to_string();
    if lyrics.is_empty() && fields.is_instrumental {
        lyrics = "[Instrumental]".to_string();
    }
    let mut lyric_lines = lyrics.lines();
    let first = lyric_lines.next().unwrap_or("");
    lines.push(format!("lyrics: {}", first));
    lines.extend(lyric_lines.map(|line| line.to_string()));
    format!("{}\n", lines.join("\n"))
}

pub fn read_carey_ace_dataset_sidecars(
    dataset_path: &str,
) -> Result<Vec<CareyAceDatasetSidecarEntry>, String> {
    let root = canonical_carey_ace_dataset_root(dataset_path)?;
    let mut files = Vec::new();
    let mut visited = std::collections::HashSet::new();
    collect_dataset_audio_files(
        &root,
        &root,
        &mut visited,
        &mut files,
        is_carey_ace_dataset_audio_file,
    )?;
    files.sort_by(|left, right| {
        left.strip_prefix(&root)
            .unwrap_or(left)
            .to_string_lossy()
            .to_lowercase()
            .cmp(
                &right
                    .strip_prefix(&root)
                    .unwrap_or(right)
                    .to_string_lossy()
                    .to_lowercase(),
            )
    });

    Ok(files
        .into_iter()
        .map(|audio_path| {
            let sidecar_path = audio_path.with_extension("txt");
            let exists = sidecar_path.is_file();
            let raw_content = if exists {
                std::fs::read_to_string(&sidecar_path)
                    .unwrap_or_default()
                    .trim_start_matches('\u{feff}')
                    .to_string()
            } else {
                String::new()
            };
            let fields = parse_carey_ace_sidecar_content(&raw_content);
            CareyAceDatasetSidecarEntry {
                relative_path: audio_path
                    .strip_prefix(&root)
                    .unwrap_or(&audio_path)
                    .to_string_lossy()
                    .to_string(),
                audio_path: user_facing_windows_path(&audio_path),
                sidecar_path: user_facing_windows_path(&sidecar_path),
                exists,
                raw_content,
                fields,
            }
        })
        .collect())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::symlink;

    #[test]
    fn linked_audio_uses_the_dataset_sidecar() {
        let fixture =
            std::env::temp_dir().join(format!("gary-linked-audio-{}", std::process::id()));
        let dataset = fixture.join("dataset");
        let source = fixture.join("source");
        fs::create_dir_all(&dataset).unwrap();
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("original.wav"), b"audio").unwrap();
        fs::write(source.join("original.txt"), "Source caption").unwrap();
        symlink("../source/original.wav", dataset.join("track.wav")).unwrap();
        fs::write(dataset.join("track.txt"), "caption: Dataset caption").unwrap();
        symlink("../source", dataset.join("outside")).unwrap();
        symlink("../source/missing.wav", dataset.join("missing.wav")).unwrap();

        let root = dataset.canonicalize().unwrap();
        let sa3 = read_sa3_dataset_sidecars(dataset.to_str().unwrap()).unwrap();
        assert_eq!(sa3.len(), 1);
        assert_eq!(sa3[0].relative_path, "track.wav");
        assert_eq!(sa3[0].content, "caption: Dataset caption");
        let carey = read_carey_ace_dataset_sidecars(dataset.to_str().unwrap()).unwrap();
        assert_eq!(carey.len(), 1);
        assert_eq!(carey[0].audio_path, sa3[0].audio_path);
        assert_eq!(carey[0].fields.caption, "Dataset caption");
        assert_eq!(
            resolve_dataset_audio_entry(
                &root,
                &dataset.join("track.wav"),
                is_sa3_dataset_audio_file
            )
            .unwrap()
            .with_extension("txt"),
            root.join("track.txt")
        );
        assert!(resolve_dataset_audio_entry(
            &root,
            &dataset.join("outside/original.wav"),
            is_sa3_dataset_audio_file
        )
        .is_err());
        symlink("../source/original.txt", dataset.join("linked.txt")).unwrap();
        assert!(check_writable_sidecar(&dataset.join("linked.txt")).is_err());
        fs::remove_dir_all(fixture).unwrap();
    }
}
