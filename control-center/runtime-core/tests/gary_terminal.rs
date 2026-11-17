#![cfg(unix)]
use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Fixture {
    root: PathBuf,
    config: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "gary-terminal-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        Self {
            config: root.join("sa3.json"),
            root,
        }
    }
    fn open(&self) -> Terminal {
        let mut master = -1;
        let mut slave = -1;
        assert_eq!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null(),
                )
            },
            0
        );
        for fd in [master, slave] {
            assert_ne!(
                unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) },
                -1
            );
        }
        let master = unsafe { File::from_raw_fd(master) };
        let slave = unsafe { File::from_raw_fd(slave) };
        let child = Command::new(env!("CARGO_BIN_EXE_gary"))
            .env("GARY4LOCAL_HEADLESS_CONFIG", &self.config)
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave))
            .spawn()
            .unwrap();
        Terminal {
            master,
            child,
            text: String::new(),
            consumed: 0,
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
struct Terminal {
    master: File,
    child: Child,
    text: String,
    consumed: usize,
}
impl Terminal {
    fn send(&mut self, line: &str) {
        self.master.write_all(line.as_bytes()).unwrap();
    }
    fn expect(&mut self, needle: &str) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(offset) = self.text[self.consumed..].find(needle) {
                self.consumed += offset + needle.len();
                return;
            }
            assert!(
                Instant::now() < deadline,
                "Waiting for {needle:?}: {}",
                self.text
            );
            let mut fd = libc::pollfd {
                fd: self.master.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            if unsafe { libc::poll(&mut fd, 1, 100) } > 0 {
                let mut bytes = [0; 4096];
                let n = self
                    .master
                    .read(&mut bytes)
                    .unwrap_or_else(|e| panic!("{e}: {}", self.text));
                assert!(n > 0, "Terminal closed before {needle}: {}", self.text);
                self.text.push_str(&String::from_utf8_lossy(&bytes[..n]));
            }
        }
    }
    fn exited(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success(), "{}", self.text);
                return;
            }
            assert!(
                Instant::now() < deadline,
                "Console did not exit: {}",
                self.text
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}
impl Drop for Terminal {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn real_terminal_setup_can_cancel_save_and_report_missing_runtime_without_downloads() {
    let fixture = Fixture::new();
    let models = fixture.root.join("existing models with spaces");
    std::fs::create_dir(&models).unwrap();
    let mut terminal = fixture.open();
    terminal.expect("not configured");
    terminal.expect("gary> ");
    terminal.send("/setup\n");
    terminal.expect("SA3 models: ");
    terminal.send("/cancel\n");
    terminal.expect("no config was saved");
    terminal.expect("gary> ");
    assert!(!fixture.config.exists());
    terminal.send("/setup\n");
    terminal.expect("SA3 models: ");
    terminal.send(&format!("{}\n", models.display()));
    terminal.expect("Client port");
    terminal.send("0\n");
    terminal.expect("Choose a port from 1 to 65535.");
    terminal.send("\n");
    terminal.expect("GPU device");
    terminal.send("\n");
    terminal.expect("Generation precision");
    terminal.send("\n");
    terminal.expect("Save [y/N]: ");
    terminal.send("y\n");
    terminal.expect("Created config");
    terminal.expect("gary> ");
    assert!(fixture.config.exists());
    assert!(!fixture.root.join("sa3.data").exists());
    terminal.send("/start\n");
    terminal.expect("Resolve the preflight checks before /start");
    terminal.expect("gary> ");
    assert!(!fixture.root.join("sa3.data").exists());
    terminal.send("/train\n");
    terminal.expect("Unknown command");
    terminal.expect("gary> ");
    terminal.send("/quit\n");
    terminal.expect("Goodbye.");
    terminal.exited();
}
#[test]
fn ctrl_c_exits_without_waiting_for_enter_or_creating_storage() {
    let fixture = Fixture::new();
    let mut terminal = fixture.open();
    terminal.expect("gary> ");
    assert_eq!(
        unsafe { libc::kill(terminal.child.id() as i32, libc::SIGINT) },
        0
    );
    terminal.expect("Goodbye.");
    terminal.exited();
    assert!(!fixture.config.exists());
    assert!(!fixture.root.join("sa3.data").exists());
}
