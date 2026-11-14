#![cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[test]
fn user_installer_quotes_paths_is_idempotent_and_refuses_clobbering() {
    let root = std::env::temp_dir().join(format!(
        "gary-installer-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _cleanup = Fixture(root.clone());
    let package = root.join("package ' $(touch INJECTED)");
    let bin = root.join("bin ' with spaces");
    let config = root.join("config ' $(touch INJECTED).json");
    std::fs::create_dir_all(&package).unwrap();
    std::fs::copy(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("install-gary.sh"),
        package.join("install-gary.sh"),
    )
    .unwrap();
    // An installer-only fixture echoes the exact environment/arguments it sees.
    std::fs::write(
        package.join("gary"),
        "#!/bin/sh\nprintf '%s\\n' \"$GARY4LOCAL_HEADLESS_CONFIG\" \"$@\"\n",
    )
    .unwrap();
    std::fs::set_permissions(package.join("gary"), std::fs::Permissions::from_mode(0o755)).unwrap();
    let install = || {
        Command::new("sh")
            .arg(package.join("install-gary.sh"))
            .arg("--bin-dir")
            .arg(&bin)
            .arg("--config")
            .arg(&config)
            .current_dir(&package)
            .output()
            .unwrap()
    };
    let first = install();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let second = install();
    assert!(second.status.success());
    assert!(String::from_utf8_lossy(&second.stdout).contains("already installed"));
    let launched = Command::new(bin.join("gary"))
        .args(["status", "a literal $argument"])
        .current_dir(&package)
        .output()
        .unwrap();
    assert!(launched.status.success());
    assert_eq!(
        String::from_utf8_lossy(&launched.stdout),
        format!("{}\nstatus\na literal $argument\n", config.display())
    );
    assert!(!package.join("INJECTED").exists());
    assert!(!config.exists());
    std::fs::write(bin.join("gary"), b"existing unrelated command").unwrap();
    assert!(!install().status.success());
    assert_eq!(
        std::fs::read(bin.join("gary")).unwrap(),
        b"existing unrelated command"
    );
}
