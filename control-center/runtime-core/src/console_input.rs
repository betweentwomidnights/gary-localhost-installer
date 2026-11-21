//! A single stdin reader serves both cooked command lines and Linux picker keys.
use std::io::Read;
use tokio::sync::mpsc;

pub(super) struct Input {
    bytes: mpsc::Receiver<u8>,
    line: Vec<u8>,
}

impl Input {
    pub(super) fn new() -> Self {
        let (tx, bytes) = mpsc::channel(256);
        // Blocking stdin must not hold up Tokio shutdown when a signal arrives.
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            let mut buf = [0; 256];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        for byte in &buf[..n] {
                            if tx.blocking_send(*byte).is_err() {
                                return;
                            }
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
        });
        Self {
            bytes,
            line: Vec::new(),
        }
    }

    // Keep the unfinished line when a ready/status event wins tokio::select!.
    pub(super) async fn recv(&mut self) -> Option<Result<String, String>> {
        loop {
            match self.bytes.recv().await {
                Some(b'\n') => break,
                Some(byte) => self.line.push(byte),
                None if self.line.is_empty() => return None,
                None => break,
            }
        }
        Some(String::from_utf8(std::mem::take(&mut self.line)).map_err(|e| e.to_string()))
    }

    #[cfg(target_os = "linux")]
    pub(super) async fn select(&mut self, title: &str, choices: &[&str]) -> Result<usize, String> {
        self.select_at(title, choices, 0).await
    }

    #[cfg(target_os = "linux")]
    pub(super) async fn select_at(
        &mut self,
        title: &str,
        choices: &[&str],
        mut selected: usize,
    ) -> Result<usize, String> {
        use super::{prompt, say};
        use std::time::Duration;
        let _mode = PickerMode::enter()?;
        say(format!(
            "{title}  (↑/↓ to choose, Enter to select, Esc to cancel)"
        ))?;
        let mut prefix = String::new();
        let draw = |index: usize| {
            let mut rows = String::new();
            for (i, label) in choices.iter().enumerate() {
                rows.push_str(if i == index { "  > " } else { "    " });
                rows.push_str(label);
                rows.push_str("\x1b[K\n");
            }
            prompt(&rows)
        };
        draw(selected)?;
        loop {
            let key = self.bytes.recv().await.ok_or("Terminal input closed.")?;
            match key {
                b'\n' | b'\r' => return Ok(selected),
                4 => return Err("Selection cancelled.".into()),
                27 => {
                    prefix.clear();
                    let next =
                        tokio::time::timeout(Duration::from_millis(120), self.bytes.recv()).await;
                    if !matches!(next, Ok(Some(b'[' | b'O'))) {
                        return Err("Selection cancelled.".into());
                    }
                    match tokio::time::timeout(Duration::from_millis(120), self.bytes.recv()).await
                    {
                        Ok(Some(b'A')) => selected = selected.saturating_sub(1),
                        Ok(Some(b'B')) => selected = (selected + 1).min(choices.len() - 1),
                        _ => continue,
                    }
                }
                byte if byte.is_ascii_alphanumeric() || byte == b'_' => {
                    prefix.push(char::from(byte.to_ascii_lowercase()));
                    if let Some(index) = choices
                        .iter()
                        .position(|label| label.to_ascii_lowercase().starts_with(&prefix))
                    {
                        selected = index;
                    } else {
                        prefix.pop();
                        continue;
                    }
                }
                _ => continue,
            }
            prompt(&format!("\x1b[{}A\r", choices.len()))?;
            draw(selected)?;
        }
    }
}

#[cfg(target_os = "linux")]
struct PickerMode(libc::termios);

#[cfg(target_os = "linux")]
impl PickerMode {
    fn enter() -> Result<Self, String> {
        let mut saved = std::mem::MaybeUninit::uninit();
        // stdin is the terminal verified by the interactive entry point.
        if unsafe { libc::tcgetattr(libc::STDIN_FILENO, saved.as_mut_ptr()) } != 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        let saved = unsafe { saved.assume_init() };
        let mut mode = saved;
        mode.c_lflag &= !(libc::ICANON | libc::ECHO);
        mode.c_cc[libc::VMIN] = 1;
        mode.c_cc[libc::VTIME] = 0;
        if unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &mode) } != 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        Ok(Self(saved))
    }
}

#[cfg(target_os = "linux")]
impl Drop for PickerMode {
    fn drop(&mut self) {
        unsafe {
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.0);
        }
    }
}
