//! Inference checkpoint selection for LoRAs published by headless training.
use crate::{
    headless::{self, Config},
    sa3_adapter::AdapterRegistry,
    service::Service,
    training::{carey, sa3},
    training_job::{adapter_path, AdapterDirectory, Scratch},
};
use std::{
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
};

const PEFT_WEIGHTS: &str = "adapter_model.safetensors";
const PEFT_CONFIG: &str = "adapter_config.json";

fn regular(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|meta| meta.is_file())
}

fn directory(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|meta| meta.is_dir())
}

pub(crate) fn registered(config: &Config) -> Result<Vec<String>, String> {
    if let Some(root) = &config.library_root {
        return Ok(crate::lora_library::catalog(root, config.service)?
            .keys()
            .cloned()
            .collect());
    }
    let adapters = config.adapters_dir();
    if !directory(&adapters) {
        return Ok(Vec::new());
    }
    if config.service == Service::Sa3 {
        return AdapterDirectory(adapters)
            .entries()
            .map(|entries| entries.into_iter().map(|entry| entry.name).collect());
    }
    let mut names = Vec::new();
    for entry in fs::read_dir(adapters).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        if let Ok(name) = entry.file_name().into_string() {
            if adapter_path(config, &name).is_ok() && published_weights(config, &name).is_ok() {
                names.push(name);
            }
        }
    }
    names.sort();
    Ok(names)
}

fn published_weights(config: &Config, name: &str) -> Result<PathBuf, String> {
    let path = adapter_path(config, name)?;
    let weights = match config.service {
        Service::Sa3 if regular(&path) => Some(path),
        Service::Carey if directory(&path) && regular(&path.join(PEFT_CONFIG)) => {
            Some(path.join(PEFT_WEIGHTS))
        }
        _ => None,
    };
    weights
        .filter(|path| regular(path))
        .ok_or_else(|| format!("LoRA {name} is no longer registered."))
}

// The published payload is the source of truth, including for profiles that
// predate the picker. A same-name import can't claim an unrelated run's history.
fn same_bytes(left: &Path, right: &Path) -> io::Result<bool> {
    let (mut left, mut right) = (File::open(left)?, File::open(right)?);
    let mut remaining = left.metadata()?.len();
    if remaining != right.metadata()?.len() {
        return Ok(false);
    }
    let (mut a, mut b) = ([0; 64 * 1024], [0; 64 * 1024]);
    while remaining > 0 {
        let count = remaining.min(a.len() as u64) as usize;
        left.read_exact(&mut a[..count])?;
        right.read_exact(&mut b[..count])?;
        if a[..count] != b[..count] {
            return Ok(false);
        }
        remaining -= count as u64;
    }
    Ok(true)
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Checkpoint {
    pub label: String,
    pub(crate) weights: PathBuf,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct History {
    pub checkpoints: Vec<Checkpoint>,
    pub active: usize,
}

fn carey_weights(checkpoint: &Path, active_config: &Path) -> Option<PathBuf> {
    if !directory(checkpoint) {
        return None;
    }
    let generation = carey::resolve_checkpoint(checkpoint)
        .ok()?
        .canonicalize()
        .ok()?;
    if !generation.starts_with(checkpoint) {
        return None;
    }
    let weights = generation.join(PEFT_WEIGHTS);
    let config = generation.join(PEFT_CONFIG);
    // All saves in a run use the same PEFT config. Keeping it and adapter_info
    // in place lets us replace just the weights in one atomic operation.
    (regular(&weights) && regular(&config) && same_bytes(&config, active_config).ok()?)
        .then_some(weights)
}

fn run_checkpoints(
    service: Service,
    run: &Path,
    progress: &sa3::Progress,
    active: &Path,
) -> Result<Vec<Checkpoint>, String> {
    let (final_path, final_label, limit) = if service == Service::Sa3 {
        (
            run.join("adapter-final.gguf"),
            format!("Final · Step {}", progress.step),
            progress.step,
        )
    } else {
        (
            run.to_owned(),
            format!("Final · Epoch {}", progress.epoch),
            progress.epoch,
        )
    };
    if Path::new(&progress.final_adapter)
        .canonicalize()
        .ok()
        .as_deref()
        != Some(final_path.as_path())
    {
        return Ok(Vec::new());
    }
    let peft_config = active.with_file_name(PEFT_CONFIG);
    let final_weights = if service == Service::Sa3 {
        regular(&final_path).then_some(final_path)
    } else {
        carey_weights(&final_path, &peft_config)
    };
    let Some(weights) = final_weights else {
        return Ok(Vec::new());
    };
    let final_checkpoint = Checkpoint {
        label: final_label,
        weights,
    };
    let mut earlier = Vec::new();
    for entry in fs::read_dir(run).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let number: Option<u32> = if service == Service::Sa3 {
            name.strip_prefix("adapter-step-")
                .and_then(|number| number.strip_suffix(".gguf"))
                .and_then(|number| number.parse().ok())
        } else {
            name.strip_prefix("checkpoint-epoch-")
                .and_then(|number| number.parse().ok())
        };
        let Some(number) = number.filter(|number| *number > 0 && *number < limit) else {
            continue;
        };
        let path = entry.path();
        let weights = if service == Service::Sa3 {
            (regular(&path) && regular(&run.join(format!("trainer-state-step-{number}.gguf"))))
                .then_some(path)
        } else {
            carey_weights(&path, &peft_config)
        };
        if let Some(weights) = weights {
            let unit = if service == Service::Sa3 {
                "Step"
            } else {
                "Epoch"
            };
            earlier.push((
                number,
                Checkpoint {
                    label: format!("{unit} {number}"),
                    weights,
                },
            ));
        }
    }
    earlier.sort_by_key(|(number, _)| std::cmp::Reverse(*number));
    Ok(std::iter::once(final_checkpoint)
        .chain(earlier.into_iter().map(|(_, checkpoint)| checkpoint))
        .collect())
}

pub(crate) fn history(config: &Config, name: &str) -> Result<Option<History>, String> {
    let active = published_weights(config, name)?;
    let training = config.training_dir();
    let jobs = training.join("jobs");
    if !directory(&training) || !directory(&jobs) {
        return Ok(None);
    }
    let mut directories = crate::dataset::read_dir_sorted(&jobs)?;
    directories.reverse();
    for job in directories {
        if !directory(&job) {
            continue;
        }
        let options = job.join("native-options.json");
        let progress_path = job.join("native-progress.json");
        if !regular(&options) || !regular(&progress_path) {
            continue;
        }
        let options = fs::read(&options).map_err(|error| error.to_string())?;
        let Ok(options) = serde_json::from_slice::<serde_json::Value>(&options) else {
            continue;
        };
        if options["name"] != name {
            continue;
        }
        let Ok(progress) = sa3::read_progress(&progress_path) else {
            continue;
        };
        let run = job.join("run");
        if !matches!(progress.status.as_str(), "completed" | "cancelled") || !directory(&run) {
            continue;
        }
        let run = run.canonicalize().map_err(|error| error.to_string())?;
        let checkpoints = run_checkpoints(config.service, &run, &progress, &active)?;
        for (index, checkpoint) in checkpoints.iter().enumerate() {
            if same_bytes(&active, &checkpoint.weights).map_err(|error| error.to_string())? {
                return Ok(Some(History {
                    checkpoints,
                    active: index,
                }));
            }
        }
    }
    Ok(None)
}

pub(crate) fn activate(
    config: &Config,
    name: &str,
    shown: &History,
    selected: usize,
) -> Result<(), String> {
    let _lease = headless::reserve_tool(config)?;
    let current = history(config, name)?.ok_or("This LoRA has no saved training history.")?;
    if &current != shown {
        return Err("This LoRA's checkpoint history changed. Open the picker again.".into());
    }
    let checkpoint = current
        .checkpoints
        .get(selected)
        .ok_or("Unknown checkpoint.")?;
    let dest = published_weights(config, name)?;
    if config.library_root.is_some() {
        return crate::lora_library::select(config, name, &checkpoint.weights);
    }
    let stage = Scratch::new(config, "checkpoint")?;
    let weights = stage.0.join("weights");
    fs::copy(&checkpoint.weights, &weights).map_err(|error| error.to_string())?;
    if config.service == Service::Sa3 {
        crate::sa3_gguf::creative_adapter(&weights)?;
    }
    fs::rename(&weights, &dest).map_err(|error| format!("Could not activate checkpoint: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn legacy_publications_can_switch_and_return_to_final() {
        for service in [Service::Sa3, Service::Carey] {
            let root =
                std::env::temp_dir().join(format!("gary-lora-history-{}", headless::serial()));
            let config: Config = serde_json::from_value(json!({
                "service": service, "runtime_root": &root, "device": "Vulkan0",
                "public_port": 8006, "model": "medium", "encoding": "f16"
            }))
            .unwrap();
            let job = root.join("training/jobs/fixture");
            let run = job.join("run");
            fs::create_dir_all(&run).unwrap();
            fs::create_dir_all(root.join("adapters")).unwrap();
            fs::create_dir_all(config.installed_dir()).unwrap();
            fs::write(root.join(".gary4local-headless"), "gary4local-server v1\n").unwrap();
            fs::write(
                config.installed_dir().join("BUILD-INFO.json"),
                json!({
                    "service": service.package(), "platform": crate::runtime::current_platform(),
                    "dirty": false, "tests_run": true, "commit": "fixture", "ggml_commit": "fixture"
                })
                .to_string(),
            )
            .unwrap();
            fs::write(job.join("native-options.json"), r#"{"name":"fixture"}"#).unwrap();
            let mut dest = adapter_path(&config, "fixture").unwrap();
            let earlier = if service == Service::Sa3 {
                let file = run.join("adapter-step-500.gguf");
                fs::write(&file, crate::sa3_gguf::fixture("sa3-lora", "dit")).unwrap();
                fs::write(run.join("trainer-state-step-500.gguf"), b"trainer state").unwrap();
                file
            } else {
                fs::create_dir(&dest).unwrap();
                fs::write(dest.join(PEFT_CONFIG), r#"{"peft_type":"LORA"}"#).unwrap();
                fs::write(
                    dest.join("adapter_info.json"),
                    r#"{"model_family":"xl","captions":["Tone"]}"#,
                )
                .unwrap();
                dest.push(PEFT_WEIGHTS);
                peft(&run.join("checkpoint-epoch-1"), b"earlier")
            };
            let (final_adapter, final_weights) = if service == Service::Sa3 {
                let final_adapter = run.join("adapter-final.gguf");
                let mut bytes = fs::read(&earlier).unwrap();
                *bytes.last_mut().unwrap() ^= 1;
                fs::write(&final_adapter, bytes).unwrap();
                (final_adapter.clone(), final_adapter)
            } else {
                (run.clone(), peft(&run, b"final"))
            };
            fs::write(
                job.join("native-progress.json"),
                json!({
                    "schema_version": 1, "status": "completed", "phase": "completed",
                    "message": "", "error": "", "step": 1000, "epoch": 2, "max_steps": 1000,
                    "final_adapter": final_adapter, "adapter_checkpoint": ""
                })
                .to_string(),
            )
            .unwrap();
            fs::copy(&final_weights, &dest).unwrap();

            assert_eq!(registered(&config).unwrap(), ["fixture"]);
            let shown = history(&config, "fixture").unwrap().unwrap();
            assert_eq!((shown.active, shown.checkpoints.len()), (0, 2));
            activate(&config, "fixture", &shown, 1).unwrap();
            assert!(same_bytes(&dest, &earlier).unwrap());
            assert!(activate(&config, "fixture", &shown, 1).is_err());
            let switched = history(&config, "fixture").unwrap().unwrap();
            assert_eq!(switched.active, 1);
            activate(&config, "fixture", &switched, 0).unwrap();
            assert!(same_bytes(&dest, &final_weights).unwrap());
            if service == Service::Carey {
                assert_eq!(
                    fs::read_to_string(dest.with_file_name("adapter_info.json")).unwrap(),
                    r#"{"model_family":"xl","captions":["Tone"]}"#
                );
            }
            fs::write(&dest, b"unrelated import").unwrap();
            assert!(history(&config, "fixture").unwrap().is_none());
            fs::remove_dir_all(root).unwrap();
        }
    }

    fn peft(checkpoint: &Path, weights: &[u8]) -> PathBuf {
        let relative = ".checkpoint-generations/generation-1";
        let generation = checkpoint.join(relative);
        fs::create_dir_all(&generation).unwrap();
        let pointer = checkpoint.join("checkpoint.current");
        #[cfg(unix)]
        std::os::unix::fs::symlink(relative, pointer).unwrap();
        #[cfg(not(unix))]
        fs::write(pointer, relative).unwrap();
        fs::write(generation.join(PEFT_CONFIG), r#"{"peft_type":"LORA"}"#).unwrap();
        let path = generation.join(PEFT_WEIGHTS);
        fs::write(&path, weights).unwrap();
        path
    }
}
