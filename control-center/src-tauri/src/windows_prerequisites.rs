//! One centrally serviced VC runtime for every native service, including CPU.
//! CUDA packs remain a separate PATH dependency. No system DLLs are copied.

#[cfg(target_os = "windows")]
use std::path::{Path, PathBuf};

#[cfg(any(test, target_os = "windows"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Version([u16; 4]);

#[cfg(any(test, target_os = "windows"))]
fn parse_version(text: &str) -> Result<Version, String> {
    let parts = text
        .trim_start_matches('v')
        .split('.')
        .map(str::parse::<u16>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| format!("invalid VC runtime version: {text}"))?;
    let values: [u16; 4] = parts
        .try_into()
        .map_err(|_| format!("invalid VC runtime version: {text}"))?;
    Ok(Version(values))
}

#[cfg(any(test, target_os = "windows"))]
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Pin {
    minimum_version: String,
    installer_version: String,
    url: String,
    sha256: String,
}

#[cfg(any(test, target_os = "windows"))]
fn pin() -> Pin {
    serde_json::from_str(include_str!("../windows/vc-runtime.json"))
        .expect("bundled VC runtime pin")
}

#[cfg(target_os = "windows")]
fn file_version(path: &Path) -> Option<Version> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileVersionInfoSizeW, GetFileVersionInfoW, VerQueryValueW, VS_FIXEDFILEINFO,
    };
    let name: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    unsafe {
        let size = GetFileVersionInfoSizeW(name.as_ptr(), std::ptr::null_mut());
        if size == 0 {
            return None;
        }
        // u64 storage provides alignment for the version resource's fixed info.
        let mut data = vec![0u64; (size as usize + 7) / 8];
        if GetFileVersionInfoW(name.as_ptr(), 0, size, data.as_mut_ptr().cast()) == 0 {
            return None;
        }
        let mut value = std::ptr::null_mut();
        let mut length = 0;
        if VerQueryValueW(
            data.as_ptr().cast(),
            [b'\\' as u16, 0].as_ptr(),
            &mut value,
            &mut length,
        ) == 0
            || length < std::mem::size_of::<VS_FIXEDFILEINFO>() as u32
            || value.is_null()
        {
            return None;
        }
        let info = std::ptr::read_unaligned(value.cast::<VS_FIXEDFILEINFO>());
        if info.dwSignature != 0xfeef04bd {
            return None;
        }
        Some(Version([
            (info.dwFileVersionMS >> 16) as u16,
            info.dwFileVersionMS as u16,
            (info.dwFileVersionLS >> 16) as u16,
            info.dwFileVersionLS as u16,
        ]))
    }
}

#[cfg(any(test, target_os = "windows"))]
const DLLS: [&str; 4] = [
    "msvcp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vcomp140.dll",
];

#[cfg(any(test, target_os = "windows"))]
fn missing_components(
    minimum: Version,
    mut version: impl FnMut(&str) -> Option<Version>,
) -> Vec<String> {
    DLLS.into_iter()
        .filter(|dll| version(dll).map_or(true, |actual| actual < minimum))
        .map(str::to_string)
        .collect()
}

/// Read-only preflight on service launch and probes. Never elevate/download as
/// a side effect of starting a service; the Install runtime action owns repair.
pub fn check() -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        let config = pin();
        let minimum = parse_version(&config.minimum_version)?;
        let system =
            PathBuf::from(std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into()))
                .join("System32");
        let missing = missing_components(minimum, |dll| file_version(&system.join(dll)));
        if !missing.is_empty() {
            return Err(format!("The shared Microsoft Visual C++ x64 runtime {} or newer is required (missing/outdated: {}). Choose Install runtime to repair it once for all native services. Windows may request administrator approval. If managed by your organisation, install Microsoft's x64 VC Redistributable first: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist", config.minimum_version, missing.join(", ")));
        }
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn hash_matches(bytes: &[u8], expected: &str) -> bool {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes)) == expected
}

#[cfg(any(test, target_os = "windows"))]
fn installer_result(code: Option<i32>, ready: bool) -> Result<(), String> {
    if code == Some(3010) || code == Some(1641) {
        return Err("Microsoft Visual C++ runtime installation requires a Windows restart. Restart Windows, then retry Install runtime.".into());
    }
    if matches!(code, Some(0 | 1638)) && ready {
        return Ok(());
    }
    Err(format!("Microsoft Visual C++ runtime installation was cancelled or failed (exit {code:?}). Retry Install runtime and approve the Windows administrator prompt, or ask your administrator to install Microsoft's x64 VC Redistributable."))
}

/// Called under native_runtime's shared installer lock, before staging any
/// service. An existing sufficiently recent system runtime is never replaced.
pub async fn ensure(cache: &std::path::Path) -> Result<(), String> {
    #[cfg(not(target_os = "windows"))]
    {
        let _ = cache;
        Ok(())
    }
    #[cfg(target_os = "windows")]
    {
        use tokio::io::AsyncWriteExt;
        if check().is_ok() {
            return Ok(());
        }
        let config = pin();
        let dir = cache.join("vc-runtime");
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| format!("Cannot create VC prerequisite cache: {e}"))?;
        let exe = dir.join(format!("VC_redist-{}-x64.exe", config.installer_version));
        let cached = tokio::fs::read(&exe).await.ok();
        if !cached
            .as_ref()
            .is_some_and(|bytes| hash_matches(bytes, &config.sha256))
        {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(180))
                .build()
                .map_err(|e| e.to_string())?;
            let mut response = client
                .get(&config.url)
                .send()
                .await
                .map_err(|e| format!("Cannot download Microsoft VC prerequisite: {e}"))?
                .error_for_status()
                .map_err(|e| e.to_string())?;
            let mut bytes = Vec::new();
            while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
                if bytes.len() + chunk.len() > 64 * 1024 * 1024 {
                    return Err("VC prerequisite download exceeds 64 MiB".into());
                }
                bytes.extend_from_slice(&chunk);
            }
            if !hash_matches(&bytes, &config.sha256) {
                return Err(
                    "Microsoft VC prerequisite SHA-256 mismatch; the installer was not executed."
                        .into(),
                );
            }
            let partial = dir.join("VC_redist.partial");
            let mut file = tokio::fs::File::create(&partial)
                .await
                .map_err(|e| e.to_string())?;
            file.write_all(&bytes).await.map_err(|e| e.to_string())?;
            file.sync_all().await.map_err(|e| e.to_string())?;
            drop(file);
            if exe.exists() {
                tokio::fs::remove_file(&exe)
                    .await
                    .map_err(|e| e.to_string())?;
            }
            tokio::fs::rename(&partial, &exe)
                .await
                .map_err(|e| e.to_string())?;
        }
        let script = dir.join("install-vc-runtime.ps1");
        tokio::fs::write(&script, include_str!("../windows/install-vc-runtime.ps1"))
            .await
            .map_err(|e| e.to_string())?;
        // Use the Windows-supplied shell, never a PATH-resolved PowerShell.
        let shell =
            PathBuf::from(std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into()))
                .join("System32/WindowsPowerShell/v1.0/powershell.exe");
        let mut command = tokio::process::Command::new(shell);
        command
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
            ])
            .arg(script)
            .arg("-Installer")
            .arg(&exe)
            .arg("-ExpectedVersion")
            .arg(&config.installer_version)
            .arg("-Install")
            .kill_on_drop(true);
        crate::hide_console_window(&mut command);
        let output = tokio::time::timeout(std::time::Duration::from_secs(600), command.output()).await
            .map_err(|_| "The VC runtime installer is still busy or waiting for Windows approval. Finish that operation before retrying Install runtime.".to_string())?
            .map_err(|e| format!("Cannot start the Microsoft VC prerequisite installer: {e}"))?;
        installer_result(output.status.code(), check().is_ok())
            .map_err(|e| format!("{e}\n{}", String::from_utf8_lossy(&output.stderr).trim()))?;
        check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vc_versions_are_numeric_and_newer_installs_are_preserved() {
        let minimum = parse_version(&pin().minimum_version).unwrap();
        let newer = parse_version(&pin().installer_version).unwrap();
        assert!(newer >= minimum);
        assert!(missing_components(minimum, |_| Some(newer)).is_empty());
        assert_eq!(missing_components(minimum, |_| None).len(), 4);
        let missing = missing_components(minimum, |dll| {
            if dll == "vcomp140.dll" {
                None
            } else {
                Some(newer)
            }
        });
        assert_eq!(missing, ["vcomp140.dll"]);
        assert_eq!(
            missing_components(minimum, |_| Some(Version([14, 9, 9999, 0]))).len(),
            4
        );
        assert!(parse_version("14.44").is_err());
    }

    #[test]
    fn vc_installer_never_turns_cancel_failure_or_reboot_into_success() {
        assert!(installer_result(Some(0), true).is_ok());
        assert!(installer_result(Some(1638), true).is_ok());
        for code in [
            Some(0),
            Some(1638),
            Some(1602),
            Some(1223),
            Some(1618),
            None,
        ] {
            assert!(installer_result(code, false).is_err());
        }
        assert!(installer_result(Some(1602), true).is_err());
        assert!(installer_result(Some(3010), true)
            .unwrap_err()
            .contains("restart"));
    }

    #[test]
    fn vc_pin_is_exact_and_direct_from_microsoft() {
        let config = pin();
        assert!(config
            .url
            .starts_with("https://download.visualstudio.microsoft.com/download/"));
        assert_eq!(config.sha256.len(), 64);
        assert!(config.sha256.bytes().all(|c| c.is_ascii_hexdigit()));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn vc_windows_version_reader_handles_absent_files() {
        assert!(file_version(Path::new("C:/nonexistent-gary-vc-test.dll")).is_none());
        let system =
            PathBuf::from(std::env::var_os("SystemRoot").unwrap()).join("System32/kernel32.dll");
        assert!(file_version(&system).is_some());
    }

    #[cfg(target_os = "windows")]
    #[test]
    #[ignore = "requires an already installed current VC runtime; never installs one"]
    fn vc_preinstalled_system_skips_repair() {
        check().unwrap();
        let cache =
            std::env::temp_dir().join(format!("gary-vc-no-download-{}", std::process::id()));
        assert!(!cache.exists());
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(ensure(&cache))
            .unwrap();
        assert!(
            !cache.exists(),
            "a ready system must not download, write files or elevate"
        );
    }
}
