#![cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// An extracted archive whose installer-only `gary` reports a version, or
/// echoes the exact environment/arguments it sees.
fn package(dir: &Path, version: &str) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::copy(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("install-gary.sh"),
        dir.join("install-gary.sh"),
    )
    .unwrap();
    std::fs::write(
        dir.join("gary"),
        format!(
            "#!/bin/sh\nif [ \"$1\" = --version ]; then echo {version}; exit 0; fi\nprintf '%s\\n' {version} \"$GARY4LOCAL_HEADLESS_CONFIG\" \"$@\"\n"
        ),
    )
    .unwrap();
    std::fs::set_permissions(dir.join("gary"), std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::write(dir.join("gary4local-server"), "#!/bin/sh\n").unwrap();
    std::fs::write(dir.join("BUILD-INFO.json"), format!("{{\"version\":\"{version}\"}}\n")).unwrap();
    std::fs::write(dir.join("README.md"), "readme\n").unwrap();
    // Something else that happens to share the extraction folder.
    std::fs::write(dir.join("unrelated.txt"), "not part of the archive\n").unwrap();
}

#[test]
fn user_installer_quotes_paths_versions_installs_and_refuses_clobbering() {
    let root = std::env::temp_dir().join(format!(
        "gary-installer-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _cleanup = Fixture(root.clone());
    let first = root.join("package ' $(touch INJECTED)");
    let bin = root.join("bin ' with spaces");
    let config = root.join("config ' $(touch INJECTED).json");
    let app = root.join("app ' $(touch INJECTED)");
    package(&first, "0.3.0");
    let install = |package: &Path, extra: &[&str]| -> Output {
        Command::new("sh")
            .arg(package.join("install-gary.sh"))
            .arg("--bin-dir")
            .arg(&bin)
            .arg("--config")
            .arg(&config)
            .arg("--app-dir")
            .arg(&app)
            .args(extra)
            .current_dir(package)
            .output()
            .unwrap()
    };
    let ok = |output: Output| {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).into_owned()
    };
    let launched = |args: &[&str]| {
        let output = Command::new(bin.join("gary"))
            .args(args)
            .current_dir(&root)
            .output()
            .unwrap();
        assert!(output.status.success());
        String::from_utf8_lossy(&output.stdout).into_owned()
    };
    let link = |name: &str| std::fs::read_link(app.join(name)).ok();

    ok(install(&first, &[]));
    assert_eq!(link("current"), Some(PathBuf::from("versions/0.3.0")));
    assert_eq!(link("previous"), None);
    assert!(app.join("versions/0.3.0/gary").is_file());
    assert!(app.join("versions/0.3.0/README.md").is_file());
    assert!(!app.join("versions/0.3.0/unrelated.txt").exists());
    assert_eq!(
        launched(&["status", "a literal $argument"]),
        format!("0.3.0\n{}\nstatus\na literal $argument\n", config.display())
    );
    assert!(ok(install(&first, &[])).contains("already installed"));

    // A newer archive installs beside the first and becomes current.
    let second = root.join("second");
    package(&second, "0.3.1");
    ok(install(&second, &[]));
    assert_eq!(link("current"), Some(PathBuf::from("versions/0.3.1")));
    assert_eq!(link("previous"), Some(PathBuf::from("versions/0.3.0")));
    assert!(launched(&[]).starts_with("0.3.1\n"));
    assert!(!root.join("INJECTED").exists());
    assert!(!first.join("INJECTED").exists());
    assert!(!config.exists());

    // An unrelated command is never replaced, and a refusal changes nothing.
    std::fs::write(bin.join("gary"), b"existing unrelated command").unwrap();
    let third = root.join("third");
    package(&third, "0.3.2");
    assert!(!install(&third, &[]).status.success());
    assert_eq!(std::fs::read(bin.join("gary")).unwrap(), b"existing unrelated command");
    assert_eq!(link("current"), Some(PathBuf::from("versions/0.3.1")));
    assert!(!app.join("versions/0.3.2").exists());

    // --replace moves it aside instead, and won't overwrite an earlier backup.
    let replaced = ok(install(&third, &["--replace"]));
    assert!(replaced.contains("Moved the old command aside"));
    assert_eq!(
        std::fs::read(bin.join("gary.previous")).unwrap(),
        b"existing unrelated command"
    );
    assert!(launched(&[]).starts_with("0.3.2\n"));
    std::fs::write(bin.join("gary"), b"another command").unwrap();
    assert!(!install(&third, &["--replace"]).status.success());
    assert_eq!(std::fs::read(bin.join("gary")).unwrap(), b"another command");
}
