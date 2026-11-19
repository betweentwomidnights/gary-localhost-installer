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
