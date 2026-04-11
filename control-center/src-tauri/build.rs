fn main() {
    // Ensure the resources directory exists so Tauri's glob patterns
    // don't fail during dev builds (staging script populates it for production).
    let resources_dir = std::path::Path::new("resources/services/manifests");
    if !resources_dir.exists() {
        std::fs::create_dir_all(resources_dir).ok();
        std::fs::write(resources_dir.join(".gitkeep"), "").ok();
    }
    let icon_path = std::path::Path::new("resources/icon.png");
    if !icon_path.exists() {
        std::fs::create_dir_all("resources").ok();
        std::fs::write(icon_path, "").ok();
    }

    tauri_build::build()
}
