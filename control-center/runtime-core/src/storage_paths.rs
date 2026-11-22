//! Shared filesystem conventions for the desktop and headless interfaces.
use std::path::{Path, PathBuf};

pub fn macos_application_support(home: &Path) -> PathBuf {
    home.join("Library/Application Support")
}

pub fn macos_app_data(home: &Path) -> PathBuf {
    macos_application_support(home).join("com.betweentwomidnights.gary4local")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macos_data_is_outside_the_application_bundle_and_old_app() {
        let home = Path::new("/Users/test-user");
        assert_eq!(macos_app_data(home), home.join("Library/Application Support/com.betweentwomidnights.gary4local"));
        assert_ne!(macos_app_data(home), home.join("Library/Application Support/gary4local"));
        assert!(!macos_app_data(home).to_string_lossy().contains(".app/"));
    }
}
