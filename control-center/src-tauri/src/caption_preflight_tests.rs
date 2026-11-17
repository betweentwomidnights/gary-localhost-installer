use super::native_sa3_autolabel_availability;
use std::path::PathBuf;

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        // Tests run in parallel in one process, and two threads can read the same
        // clock value, which put two fixtures in one folder: the VAE test's models
        // then showed up in the LM-only test. The counter keeps every fixture apart.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        Self(
            std::env::temp_dir().join(format!(
                "gary-caption-preflight-{}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            )),
        )
    }
    fn component(&self, id: &str, only_vae: bool) {
        let models = crate::carey_models::models();
        let directory = models.models_dir(&self.0);
        std::fs::create_dir_all(&directory).unwrap();
        for expected in &models.component(id).unwrap().files {
            if only_vae && expected.filename != crate::carey_caption::vae_file().unwrap() {
                continue;
            }
            let path = directory.join(&expected.filename);
            let file = std::fs::File::create(&path).unwrap();
            // Model-sized placeholders must not allocate gigabytes on disk.
            #[cfg(windows)]
            mark_sparse(&file);
            file.set_len(expected.bytes).unwrap();
        }
    }
}
/// Marks the file sparse on its open handle (FSCTL_SET_SPARSE), without starting
/// an `fsutil` process per file.
#[cfg(windows)]
fn mark_sparse(file: &std::fs::File) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::{Ioctl::FSCTL_SET_SPARSE, IO::DeviceIoControl};
    let mut returned = 0u32;
    let ok = unsafe {
        DeviceIoControl(
            file.as_raw_handle(),
            FSCTL_SET_SPARSE,
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            0,
            &mut returned,
            std::ptr::null_mut(),
        )
    };
    assert!(ok != 0, "FSCTL_SET_SPARSE failed: {}", std::io::Error::last_os_error());
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn lm_only_install_names_missing_audio_models_without_requesting_lm_again() {
    let fixture = Fixture::new();
    fixture.component("carey-native::lm-1.7B::BF16", false);
    let state = native_sa3_autolabel_availability(&fixture.0, "acestep-5Hz-lm-1.7B");
    assert!(state.captioner_downloaded && !state.available);
    let blocker = state.blocker.unwrap();
    assert!(blocker.contains("audio model (DiT)") && blocker.contains("shared VAE"));
    assert!(!blocker.contains("1.7B captioner"));
    assert!(
        !native_sa3_autolabel_availability(&fixture.0, "acestep-5Hz-lm-4B").captioner_downloaded
    );
}

#[test]
fn complete_vae_is_required_but_shared_text_encoder_is_not() {
    let fixture = Fixture::new();
    fixture.component("carey-native::lm-1.7B::BF16", false);
    fixture.component("carey-native::base::Q8_0", false);
    let missing = native_sa3_autolabel_availability(&fixture.0, "acestep-5Hz-lm-1.7B");
    assert!(!missing.available && missing.captioner_downloaded);
    assert!(missing.blocker.unwrap().contains("shared VAE"));
    fixture.component(crate::carey_models::SHARED, true);
    let ready = native_sa3_autolabel_availability(&fixture.0, "acestep-5Hz-lm-1.7B");
    assert!(ready.available && ready.captioner_downloaded && ready.blocker.is_none());
    let vae = crate::carey_models::models()
        .models_dir(&fixture.0)
        .join(crate::carey_caption::vae_file().unwrap());
    std::fs::OpenOptions::new()
        .write(true)
        .open(vae)
        .unwrap()
        .set_len(1)
        .unwrap();
    assert!(crate::carey_caption::caption_vae(&fixture.0).is_err());
    assert!(!native_sa3_autolabel_availability(&fixture.0, "acestep-5Hz-lm-1.7B").available);
}

#[test]
#[ignore = "read-only inventory of an explicitly selected existing model store"]
fn real_caption_inventory() {
    let root = PathBuf::from(
        std::env::var_os("GARY4LOCAL_CAPTION_SMOKE_ROOT")
            .expect("set GARY4LOCAL_CAPTION_SMOKE_ROOT"),
    );
    assert!(root.is_absolute() && root.is_dir());
    let state = native_sa3_autolabel_availability(&root, "acestep-5Hz-lm-1.7B");
    println!(
        "CAPTION_PREFLIGHT {}",
        serde_json::to_string(&state).unwrap()
    );
    assert!(
        state.captioner_downloaded,
        "the selected real LM must be installed"
    );
    if let Ok(expected) = std::env::var("GARY4LOCAL_CAPTION_EXPECT_AVAILABLE") {
        assert_eq!(state.available, expected.parse::<bool>().unwrap());
    }
}
