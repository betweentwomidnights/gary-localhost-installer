//! Tell which Carey model a PEFT adapter folder was trained on, so an adapter
//! added by hand (including ones the retired MLX gary4local for Mac trained)
//! starts out on the right model instead of failing to load on the other one.

use serde_json::Value;
use std::{fs::File, io::Read, path::Path};

/// Projections whose input is the decoder's hidden state: their `lora_A` is
/// `[rank, hidden]`, and hidden is 2048 on the 2B models and 2560 on XL.
const HIDDEN_INPUT_MODULES: [&str; 6] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "cross_attn.q_proj",
];

/// "standard" or "xl", or None when the folder doesn't say.
///
/// The adapter's own shapes decide when they can, since they can't be wrong.
/// Otherwise a trainer's sidecar does: `metadata.json` (the MLX Mac app) or
/// `adapter_info.json` (gary4local's library), under `model_family`.
pub fn detect_model_family(dir: &Path) -> Option<&'static str> {
    let from_shapes = hidden_size(&dir.join("adapter_model.safetensors")).and_then(|width| match width {
        2048 => Some("standard"),
        2560 => Some("xl"),
        _ => None,
    });
    from_shapes.or_else(|| {
        ["metadata.json", "adapter_info.json"].iter().find_map(|name| {
            let value: Value = serde_json::from_slice(&std::fs::read(dir.join(name)).ok()?).ok()?;
            ["model_family", "modelFamily"]
                .iter()
                .find_map(|key| value[key].as_str())
                .and_then(normalize)
        })
    })
}

fn normalize(family: &str) -> Option<&'static str> {
    match family.trim().to_ascii_lowercase().as_str() {
        "xl" => Some("xl"),
        "standard" | "2b" => Some("standard"),
        _ => None,
    }
}

/// The hidden width every matching `lora_A` agrees on, read from the
/// safetensors header alone.
fn hidden_size(path: &Path) -> Option<i64> {
    let mut file = File::open(path).ok()?;
    let mut size = [0u8; 8];
    file.read_exact(&mut size).ok()?;
    let size = u64::from_le_bytes(size);
    if size > 64 << 20 {
        return None;
    }
    let mut header = vec![0u8; size as usize];
    file.read_exact(&mut header).ok()?;
    let header: Value = serde_json::from_slice(&header).ok()?;
    let mut widths = header
        .as_object()?
        .iter()
        .filter(|(key, _)| is_hidden_input_lora_a(key))
        .map(|(_, tensor)| tensor["shape"].as_array().and_then(|shape| shape.get(1)?.as_i64()));
    let first = widths.next()??;
    widths.all(|width| width == Some(first)).then_some(first)
}

/// "<...>.<module>.lora_A.weight" or "<...>.<module>.lora_A.<adapter>.weight".
fn is_hidden_input_lora_a(key: &str) -> bool {
    let Some((module, rest)) = key.split_once(".lora_A") else {
        return false;
    };
    let adapter = rest.strip_suffix(".weight");
    let named = matches!(adapter, Some("")) || adapter.is_some_and(|a| a.len() > 1 && a.starts_with('.') && !a[1..].contains('.'));
    named && HIDDEN_INPUT_MODULES.iter().any(|m| module.ends_with(&format!(".{m}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Scratch(std::path::PathBuf);
    impl Scratch {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("gary-carey-family-{}", crate::headless::serial()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn adapter(dir: &Path, hidden: &[i64], adapter_name: &str) {
        std::fs::create_dir_all(dir).unwrap();
        let mut header = serde_json::Map::new();
        for (layer, width) in hidden.iter().enumerate() {
            for (module, input) in [("self_attn.q_proj", *width), ("self_attn.o_proj", 4096), ("mlp.down_proj", 6144)] {
                for (kind, shape) in [("lora_A", json!([16, input])), ("lora_B", json!([input, 16]))] {
                    header.insert(
                        format!("base_model.model.layers.{layer}.{module}.{kind}{adapter_name}.weight"),
                        json!({"dtype":"F16","shape":shape,"data_offsets":[0,0]}),
                    );
                }
            }
        }
        header.insert("__metadata__".into(), json!({"format":"pt"}));
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        std::fs::write(dir.join("adapter_model.safetensors"), bytes).unwrap();
        std::fs::write(dir.join("adapter_config.json"), r#"{"peft_type":"LORA"}"#).unwrap();
    }

    #[test]
    fn shapes_name_the_family() {
        let root = Scratch::new();
        adapter(&root.path().join("std"), &[2048; 24], ".default");
        adapter(&root.path().join("xl"), &[2560; 32], "");
        adapter(&root.path().join("odd"), &[1536; 4], ".default");
        adapter(&root.path().join("mixed"), &[2048, 2560], ".default");
        assert_eq!(detect_model_family(&root.path().join("std")), Some("standard"));
        assert_eq!(detect_model_family(&root.path().join("xl")), Some("xl"));
        assert_eq!(detect_model_family(&root.path().join("odd")), None);
        assert_eq!(detect_model_family(&root.path().join("mixed")), None);
        assert_eq!(detect_model_family(&root.path().join("missing")), None);
    }

    #[test]
    fn a_sidecar_covers_what_shapes_cannot() {
        let root = Scratch::new();
        let odd = root.path().join("odd");
        adapter(&odd, &[1536; 4], ".default");
        std::fs::write(odd.join("metadata.json"), r#"{"modelFamily":"xl"}"#).unwrap();
        assert_eq!(detect_model_family(&odd), Some("xl"));

        let library = root.path().join("library");
        std::fs::create_dir_all(&library).unwrap();
        std::fs::write(library.join("adapter_info.json"), r#"{"model_family":"standard"}"#).unwrap();
        assert_eq!(detect_model_family(&library), Some("standard"));

        // A sidecar that disagrees with the weights loses: the weights are what load.
        let xl = root.path().join("xl");
        adapter(&xl, &[2560; 32], ".default");
        std::fs::write(xl.join("metadata.json"), r#"{"model_family":"standard"}"#).unwrap();
        assert_eq!(detect_model_family(&xl), Some("xl"));
    }
}
