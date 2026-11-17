//! Renderer compatibility is separate from CUDA/Vulkan generation backends.
use std::path::Path;

fn needs_nvidia_x11_workaround(
    nvidia_driver: bool,
    session: Option<&str>,
    display: bool,
    wayland: bool,
    user_override: bool,
) -> bool {
    !user_override
        && nvidia_driver
        && (session == Some("x11") || (session.is_none() && display && !wayland))
}

/// Called before Tauri/GTK or worker threads are initialized. On the Spark's
/// NVIDIA/X11 session, disabling the WebKit DMA-BUF path fixes a blank webview.
/// Leave working AMD/Intel/Wayland setups and explicit user choices alone.
/// https://v2.tauri.app/develop/debug/linux-graphics/
pub fn configure_renderer() {
    let session = std::env::var("XDG_SESSION_TYPE").ok();
    if needs_nvidia_x11_workaround(
        Path::new("/proc/driver/nvidia/version").is_file(),
        session.as_deref(),
        std::env::var_os("DISPLAY").is_some(),
        std::env::var_os("WAYLAND_DISPLAY").is_some(),
        std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_some(),
    ) {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
        eprintln!("gary4local: applying NVIDIA/X11 WebKit DMA-BUF compatibility rendering");
    }
}

#[cfg(test)]
mod tests {
    use super::needs_nvidia_x11_workaround as needs;

    #[test]
    fn nvidia_x11_selects_the_verified_workaround() {
        assert!(needs(true, Some("x11"), true, false, false));
        assert!(needs(true, None, true, false, false));
    }

    #[test]
    fn other_gpus_sessions_and_user_overrides_are_preserved() {
        assert!(!needs(false, Some("x11"), true, false, false));
        assert!(!needs(true, Some("wayland"), true, true, false));
        assert!(!needs(true, None, true, true, false));
        assert!(!needs(true, None, false, false, false));
        assert!(!needs(true, Some("x11"), true, false, true));
    }
}
