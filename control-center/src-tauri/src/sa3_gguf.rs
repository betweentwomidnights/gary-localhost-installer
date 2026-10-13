//! Bounded GGUF adapter metadata inspection before admitting a local import.
//! KV types follow the bundled sa3.cpp/ggml/include/gguf.h format; tensor loading
//! and compatibility with the selected model remain the native runtime's job.
use std::io::{Read, Take};
use std::path::Path;

struct Reader(Take<std::io::BufReader<std::fs::File>>);
impl Reader {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let mut bytes = [0; N];
        self.0
            .read_exact(&mut bytes)
            .map_err(|_| "Truncated or oversized GGUF metadata")?;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn text(&mut self) -> Result<String, String> {
        let length = self.u64()?;
        if length > 1024 * 1024 {
            return Err("GGUF metadata string is too large".into());
        }
        let mut bytes = vec![0; length as usize];
        self.0
            .read_exact(&mut bytes)
            .map_err(|_| "Truncated GGUF metadata string")?;
        String::from_utf8(bytes).map_err(|_| "GGUF metadata is not UTF-8".into())
    }
    fn skip(&mut self, length: u64) -> Result<(), String> {
        let copied = std::io::copy(&mut self.0.by_ref().take(length), &mut std::io::sink())
            .map_err(|error| error.to_string())?;
        if copied != length {
            return Err("Truncated or oversized GGUF metadata".into());
        }
        Ok(())
    }
    fn value(&mut self, kind: u32) -> Result<(), String> {
        match kind {
            8 => {
                self.text()?;
                Ok(())
            }
            9 => {
                let item = self.u32()?;
                let count = self.u64()?;
                if item == 9 || count > 65536 {
                    return Err("Invalid or oversized GGUF metadata array".into());
                }
                for _ in 0..count {
                    self.value(item)?;
                }
                Ok(())
            }
            0 | 1 | 7 => self.skip(1),
            2 | 3 => self.skip(2),
            4 | 5 | 6 => self.skip(4),
            10 | 11 | 12 => self.skip(8),
            _ => Err("Unknown GGUF metadata type".into()),
        }
    }
}

pub fn creative_adapter(path: &Path) -> Result<(), String> {
    let file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let file_bytes = file.metadata().map_err(|error| error.to_string())?.len();
    let mut reader = Reader(std::io::BufReader::new(file).take(16 * 1024 * 1024));
    if reader.bytes::<4>()? != *b"GGUF" || !matches!(reader.u32()?, 2 | 3) {
        return Err("Choose a supported GGUF LoRA adapter".into());
    }
    let tensors = reader.u64()?;
    if tensors == 0 || tensors > 65536 {
        return Err("Invalid GGUF adapter tensor count".into());
    }
    let count = reader.u64()?;
    if count == 0 || count > 4096 {
        return Err("Invalid GGUF metadata count".into());
    }
    let mut architecture = None;
    let mut target = None;
    let mut alignment = 32u64;
    let mut keys = std::collections::BTreeSet::new();
    for _ in 0..count {
        let key = reader.text()?;
        if !keys.insert(key.clone()) {
            return Err("Duplicate GGUF metadata key".into());
        }
        let kind = reader.u32()?;
        if key == "general.architecture" || key == "lora.target" {
            if kind != 8 {
                return Err("Invalid GGUF adapter identity".into());
            }
            let value = reader.text()?;
            if key == "general.architecture" {
                architecture = Some(value);
            } else {
                target = Some(value);
            }
        } else if key == "general.alignment" {
            if kind != 4 {
                return Err("Invalid GGUF alignment".into());
            }
            alignment = reader.u32()? as u64;
            if !alignment.is_power_of_two() || alignment > 4096 {
                return Err("Invalid GGUF alignment".into());
            }
        } else {
            reader.value(kind)?;
        }
    }
    if architecture.as_deref() != Some("sa3-lora") {
        return Err("This GGUF is not an SA3 LoRA adapter".into());
    }
    // Older sa3.cpp adapters omitted target for DiT, matching its native default.
    if target.as_deref().unwrap_or("dit") != "dit" {
        return Err(
            "Creative LoRAs must target the DiT. Prepare decoder correction in SA3 Models.".into(),
        );
    }
    let mut bounds = Vec::new();
    let mut names = std::collections::BTreeSet::new();
    for _ in 0..tensors {
        let name = reader.text()?;
        if name.is_empty() || !names.insert(name) {
            return Err("Invalid or duplicate GGUF tensor name".into());
        }
        let dimensions = reader.u32()?;
        if !(1..=4).contains(&dimensions) {
            return Err("Invalid GGUF tensor dimensions".into());
        }
        let mut elements = 1u64;
        for _ in 0..dimensions {
            let size = reader.u64()?;
            if size == 0 {
                return Err("Empty GGUF adapter tensor".into());
            }
            elements = elements
                .checked_mul(size)
                .ok_or("GGUF tensor dimensions overflow")?;
        }
        let kind = reader.u32()?;
        // ggml validates quantized type/shape compatibility. Check full scalar
        // F32/F16 payload bounds here without loading any tensor into memory.
        let minimum = match kind {
            0 => elements.checked_mul(4),
            1 => elements.checked_mul(2),
            _ => Some(1),
        }
        .ok_or("GGUF tensor size overflows")?;
        let offset = reader.u64()?;
        if offset % alignment != 0 {
            return Err("Invalid GGUF tensor alignment".into());
        }
        bounds.push(
            offset
                .checked_add(minimum)
                .ok_or("GGUF tensor offset overflows")?,
        );
    }
    let consumed = 16 * 1024 * 1024 - reader.0.limit();
    let start = consumed.div_ceil(alignment) * alignment;
    if bounds
        .into_iter()
        .any(|end| start.checked_add(end).is_none_or(|end| end > file_bytes))
    {
        return Err("Truncated GGUF adapter tensor data".into());
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn fixture(architecture: &str, target: &str) -> Vec<u8> {
    let mut data = b"GGUF".to_vec();
    data.extend(3u32.to_le_bytes());
    data.extend(1u64.to_le_bytes());
    data.extend(2u64.to_le_bytes());
    for (key, value) in [
        ("general.architecture", architecture),
        ("lora.target", target),
    ] {
        data.extend((key.len() as u64).to_le_bytes());
        data.extend(key.as_bytes());
        data.extend(8u32.to_le_bytes());
        data.extend((value.len() as u64).to_le_bytes());
        data.extend(value.as_bytes());
    }
    let tensor = "test.lora_A.weight";
    data.extend((tensor.len() as u64).to_le_bytes());
    data.extend(tensor.as_bytes());
    data.extend(1u32.to_le_bytes());
    data.extend(1u64.to_le_bytes());
    data.extend(0u32.to_le_bytes());
    data.extend(0u64.to_le_bytes());
    data.resize(data.len().div_ceil(32) * 32, 0);
    data.extend(1f32.to_le_bytes());
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn imports_reject_models_auxiliary_adapters_and_truncated_payloads() {
        let path =
            std::env::temp_dir().join(format!("sa3-gguf-metadata-{}.gguf", std::process::id()));
        for (architecture, target, accepted) in [
            ("sa3-lora", "dit", true),
            ("sa3-lora", "decoder", false),
            ("sa3-dit", "dit", false),
        ] {
            let mut data = fixture(architecture, target);
            std::fs::write(&path, &data).unwrap();
            assert_eq!(creative_adapter(&path).is_ok(), accepted);
            data.truncate(data.len() - 1);
            std::fs::write(&path, &data).unwrap();
            assert!(creative_adapter(&path).is_err());
        }
        let mut data = fixture("sa3-lora", "dit");
        data[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, data).unwrap();
        assert!(creative_adapter(&path).unwrap_err().contains("count"));
        std::fs::remove_file(path).unwrap();
    }
}
