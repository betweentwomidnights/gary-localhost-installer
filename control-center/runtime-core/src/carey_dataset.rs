//! Stage a Carey training dataset in the native trainer format.
use serde_json::json;
use std::path::{Path, PathBuf};

/// One sidecar's value: trimmed, on one line, as ace-train's parser reads it.
pub fn scalar(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A beats-per-bar numerator from "4", "4/4" or "6/8"; ACE-Step's own default
/// of four when the sidecar has none.
fn signature(value: &str) -> Result<u32, String> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(4);
    }
    value
        .split('/')
        .next()
        .and_then(|numerator| numerator.trim().parse::<u32>().ok())
        .filter(|beats| (1..=32).contains(beats))
        .ok_or_else(|| format!("time signature \"{value}\" is not like 4/4"))
}

fn bpm(value: &str) -> Result<u32, String> {
    let value = value.trim();
    value
        .parse::<f64>()
        .ok()
        .filter(|bpm| bpm.is_finite() && *bpm >= 1.0 && *bpm <= 1000.0)
        .map(|bpm| bpm.round() as u32)
        .ok_or_else(|| {
            if value.is_empty() {
                "no BPM".into()
            } else {
                format!("BPM \"{value}\" is not a number")
            }
        })
}

/// One Carey sidecar as ace-train metadata. ace-train requires every field
/// to be present, lyrics last; Carey writes `keyscale` and `timesignature` and
/// omits empty fields. The trigger, when set, is every example's custom tag,
/// as the Python job writes it.
pub fn metadata(
    fields: &crate::dataset::CareyAceSidecarFields,
    trigger: &str,
) -> Result<String, String> {
    let caption = scalar(&fields.caption);
    if caption.is_empty() {
        return Err("no caption".into());
    }
    let bpm = bpm(&fields.bpm)?;
    let beats = signature(&fields.timesignature)?;
    // Empty lyrics on a vocal song train as empty lyric conditioning, as in
    // PyTorch Carey; only an instrumental gets the [Instrumental] marker.
    let mut lyrics = fields.lyrics.trim().replace("\r\n", "\n");
    if lyrics.is_empty() && fields.is_instrumental {
        lyrics = "[Instrumental]".into();
    }
    let tag = if trigger.trim().is_empty() {
        scalar(&fields.custom_tag)
    } else {
        scalar(trigger)
    };
    Ok(format!(
        "caption: {caption}\ngenre: {}\nbpm: {bpm}\nkey: {}\nsignature: {beats}\nis_instrumental: {}\ncustom_tag: {tag}\nlyrics: {lyrics}\n",
        scalar(&fields.genre),
        scalar(&fields.keyscale),
        fields.is_instrumental,
    ))
}

/// Whether ace-train reads this WAV as it is: 16-bit PCM or 32-bit float.
fn readable_wav(path: &Path) -> bool {
    hound::WavReader::open(path).is_ok_and(|reader| {
        let spec = reader.spec();
        matches!(
            (spec.sample_format, spec.bits_per_sample),
            (hound::SampleFormat::Int, 16) | (hound::SampleFormat::Float, 32)
        )
    })
}

async fn stage_audio(source: &Path, dest: &Path) -> Result<(), String> {
    let wav = source
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"));
    if wav && readable_wav(source) {
        // Training reads its inputs without changing them, so a link is enough.
        if std::fs::hard_link(source, dest).is_ok() {
            return Ok(());
        }
        return std::fs::copy(source, dest)
            .map(|_| ())
            .map_err(|error| error.to_string());
    }
    let mut command = tokio::process::Command::new("ffmpeg");
    #[cfg(windows)]
    command.creation_flags(0x08000000);
    command
        .args(["-nostdin", "-v", "error", "-y", "-i"])
        .arg(source)
        .args(["-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-f", "wav"])
        .arg(dest)
        .kill_on_drop(true);
    let output = command
        .output()
        .await
        .map_err(|error| format!("FFmpeg is needed to train on {}: {error}", source.display()))?;
    if !output.status.success() {
        return Err(format!(
            "Could not decode {}: {}",
            source.display(),
            String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(500)
                .collect::<String>()
        ));
    }
    Ok(())
}

/// What staging produced: the flat dataset ace-train reads and the prompts
/// Carey's caption pool takes from it.
pub struct Staged {
    pub count: usize,
    pub captions: Vec<String>,
}

/// Add one song's prompts to the dice pool, as Python Carey's
/// build_captions.py did: its caption, then its genre line with every 2- and
/// 3-tag subset of that line. Training prompts with the genre at the genre
/// ratio, so the short prompts are ones the LoRA learned too. Subsets never
/// mix tags from different songs.
pub fn add_prompts(pool: &mut Vec<String>, caption: &str, genre: &str) {
    let tags: Vec<&str> = genre
        .split(',')
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .collect();
    let mut prompts = vec![caption.trim().to_string()];
    if !tags.is_empty() {
        prompts.push(tags.join(", "));
    }
    for size in [2, 3] {
        if tags.len() < size {
            continue;
        }
        // Every subset of `size` tags, in the line's order.
        let mut picks: Vec<usize> = (0..size).collect();
        loop {
            prompts.push(
                picks
                    .iter()
                    .map(|&index| tags[index])
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            let Some(slot) = (0..size)
                .rev()
                .find(|&slot| picks[slot] < tags.len() - size + slot)
            else {
                break;
            };
            picks[slot] += 1;
            for next in slot + 1..size {
                picks[next] = picks[next - 1] + 1;
            }
        }
    }
    for prompt in prompts {
        if !prompt.is_empty() && !pool.contains(&prompt) {
            pool.push(prompt);
        }
    }
}

/// "no BPM: a.wav, b.wav and 3 more; no caption: c.wav"
fn problem_summary(problems: &[(String, Vec<String>)]) -> String {
    problems
        .iter()
        .map(|(kind, songs)| {
            let shown = songs.iter().take(2).cloned().collect::<Vec<_>>().join(", ");
            match songs.len() {
                0..=2 => format!("{kind}: {shown}"),
                more => format!("{kind}: {shown} and {} more", more - 2),
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Stage `dataset` into `dest` as numbered WAV and metadata pairs. Every
/// sidecar problem is reported at once, before any audio is decoded.
pub async fn stage(dataset: &Path, dest: &Path, trigger: &str) -> Result<Staged, String> {
    let entries = crate::dataset::read_carey_ace_dataset_sidecars(&dataset.to_string_lossy())?;
    if entries.is_empty() {
        return Err(format!(
            "{} has no audio files to train on.",
            dataset.display()
        ));
    }
    // Problems grouped by kind, each naming a few songs: one short line.
    let mut problems: Vec<(String, Vec<String>)> = Vec::new();
    let mut staged = Vec::new();
    for entry in &entries {
        match metadata(&entry.fields, trigger) {
            Ok(text) => staged.push((
                PathBuf::from(&entry.audio_path),
                text,
                scalar(&entry.fields.caption),
                scalar(&entry.fields.genre),
            )),
            Err(problem) => match problems.iter_mut().find(|(kind, _)| *kind == problem) {
                Some((_, songs)) => songs.push(entry.relative_path.clone()),
                None => problems.push((problem, vec![entry.relative_path.clone()])),
            },
        }
    }
    if !problems.is_empty() {
        return Err(format!(
            "Fix these sidecars in the editor first: {}.",
            problem_summary(&problems)
        ));
    }
    std::fs::create_dir_all(dest).map_err(|error| error.to_string())?;
    let mut manifest = Vec::new();
    let mut captions = Vec::new();
    for (index, (audio, text, caption, genre)) in staged.iter().enumerate() {
        let stem = format!("{index:04}");
        stage_audio(audio, &dest.join(format!("{stem}.wav"))).await?;
        std::fs::write(dest.join(format!("{stem}.txt")), text)
            .map_err(|error| error.to_string())?;
        manifest.push(json!({"staged": stem, "source": audio}));
        add_prompts(&mut captions, caption, genre);
    }
    crate::headless::write_json(&dest.join("staging.json"), &manifest)?;
    Ok(Staged {
        count: staged.len(),
        captions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn large_sidecar_problem_groups_stay_readable() {
        let summary = problem_summary(&[
            (
                "no BPM".into(),
                vec![
                    "a.wav".into(),
                    "b.wav".into(),
                    "c.wav".into(),
                    "d.wav".into(),
                ],
            ),
            ("no caption".into(), vec!["e.wav".into()]),
        ]);
        assert_eq!(
            summary,
            "no BPM: a.wav, b.wav and 2 more; no caption: e.wav"
        );
    }
}
