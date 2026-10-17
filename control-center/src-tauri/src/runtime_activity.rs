//! A short journal of what happened to a shared native runtime: installs and
//! updates, each service's switch to it, and the Python installs it retired.
//! The services sharing a runtime show it together, since none of them owns
//! those events alone. One JSON line per event, newest last, in the profile's
//! logs folder.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

/// The journal keeps this many events; older ones are dropped on write.
const KEEP: usize = 200;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Event {
    /// Seconds since the Unix epoch.
    pub at: u64,
    pub text: String,
    #[serde(default)]
    pub failed: bool,
}

fn path(root: &Path, runtime: &str) -> PathBuf {
    root.join("logs").join(format!("runtime-{runtime}.jsonl"))
}

/// Record an event for `runtime` (a native bundle, or a service's own
/// runtime). Losing a journal line never fails what it describes.
pub fn record(root: &Path, runtime: &str, text: &str, failed: bool) {
    let event = Event {
        at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        text: text.to_string(),
        failed,
    };
    let path = path(root, runtime);
    let mut events = recent(root, runtime, KEEP - 1);
    events.push(event);
    let write = || -> std::io::Result<()> {
        std::fs::create_dir_all(path.parent().unwrap_or(root))?;
        let mut file = std::fs::File::create(&path)?;
        for event in &events {
            serde_json::to_writer(&mut file, event)?;
            file.write_all(b"\n")?;
        }
        Ok(())
    };
    if let Err(error) = write() {
        log::warn!("Could not record runtime activity in {}: {error}", path.display());
    }
}

/// The newest `limit` events, oldest first. Unreadable lines are skipped.
pub fn recent(root: &Path, runtime: &str, limit: usize) -> Vec<Event> {
    let Ok(text) = std::fs::read_to_string(path(root, runtime)) else {
        return Vec::new();
    };
    let events: Vec<Event> = text
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    events[events.len().saturating_sub(limit)..].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_journal_keeps_the_newest_events_in_order() {
        let root = std::env::temp_dir().join(format!("gary-runtime-activity-{}", std::process::id()));
        assert!(recent(&root, "sa3", 10).is_empty());
        for index in 0..KEEP + 5 {
            record(&root, "sa3", &format!("event {index}"), index == 3);
        }
        let events = recent(&root, "sa3", KEEP + 50);
        assert_eq!(events.len(), KEEP);
        assert_eq!(events.last().unwrap().text, format!("event {}", KEEP + 4));
        assert_eq!(recent(&root, "sa3", 2)[0].text, format!("event {}", KEEP + 3));
        assert!(recent(&root, "yuey", 10).is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }
}
