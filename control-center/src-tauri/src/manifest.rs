//! Manifest schema shared with the GUI-free installer.
pub use gary4local_core::manifest::*;

#[cfg(test)]
mod tests {
    #[test]
    fn linux_previews_pin_gpu_packages_for_both_architectures() {
        let mut manifest: super::Manifest = serde_json::from_str(super::BUNDLED_JSON).unwrap();
        manifest.resolve_native_bundles().unwrap();
        for service in &manifest.services {
            let Some(native) = service.native.as_ref() else { continue; };
            for platform in ["linux-x64", "linux-arm64"] {
                let packages = &native.platforms[platform];
                assert_eq!(packages.prefer, ["vulkan"]);
                assert_eq!(packages.backends.len(), 1);
                for (part, package) in [("core", &packages.core), ("vulkan", &packages.backends["vulkan"])] {
                    assert!(crate::native_runtime::is_sha256(&package.sha256));
                    assert!(package.requires.is_empty());
                    assert!(package.url.starts_with("https://github.com/betweentwomidnights/"));
                    assert!(package.url.contains(&format!("/releases/download/linux-{}-preview.1/", native.version)));
                    assert!(package.url.ends_with(&format!("-{platform}-{part}.zip")));
                }
            }
        }
    }
    use super::{Manifest, ServiceRuntime};

    fn bundled() -> Manifest {
        let mut manifest: Manifest =
            serde_json::from_str(include_str!("../../../services/manifests/services.json"))
                .expect("bundled service manifest should be valid JSON");
        manifest
            .resolve_native_bundles()
            .expect("bundle references should resolve");
        manifest
    }

    #[test]
    fn sa3_candidate_uses_one_pinned_bundle_without_switching_python() {
        let mut manifest = bundled();
        let sa3 = manifest
            .services
            .iter()
            .find(|service| service.id == "sa3")
            .unwrap();
        assert_eq!(sa3.runtime, ServiceRuntime::Python);
        let native = sa3.native.as_ref().unwrap();
        assert_eq!(native.bundle.as_deref(), Some("sa3"));
        assert_eq!(native.version, manifest.native_bundles["sa3"].version);
        let cuda = &native.platforms["windows-x64"].backends["cuda"];
        assert_eq!(cuda.requires, ["cudart-12.8"]);
        assert!(crate::native_runtime::is_sha256(&cuda.sha256));
        manifest.resolve_native_bundles().unwrap();
    }

    #[test]
    fn unknown_or_escaping_bundle_references_are_rejected() {
        for id in ["missing", "../sa3", "C:/sa3", "COM1", "con", "lpt2"] {
            let mut manifest = bundled();
            let native = manifest
                .services
                .iter_mut()
                .find(|service| service.id == "sa3")
                .unwrap()
                .native
                .as_mut()
                .unwrap();
            native.bundle = Some(id.to_string());
            assert!(manifest.resolve_native_bundles().is_err(), "{id}");
        }
    }

    #[test]
    fn consumer_cannot_override_shared_package_identity() {
        let mut manifest = bundled();
        manifest
            .services
            .iter_mut()
            .find(|service| service.id == "sa3")
            .unwrap()
            .native
            .as_mut()
            .unwrap()
            .version = "v9.9.9".to_string();
        assert!(manifest.resolve_native_bundles().is_err());
    }

    #[test]
    fn a_service_starts_on_python_until_it_migrates() {
        let manifest = bundled();
        let gary = manifest
            .services
            .iter()
            .find(|service| service.id == "gary")
            .expect("bundled manifest should define gary");

        // Its native block is what it can migrate to, not how it starts.
        assert_eq!(gary.runtime, ServiceRuntime::Python);
        assert_eq!(gary.entry_point, "g4l_localhost.py");
        assert_eq!(
            gary.native
                .as_ref()
                .and_then(|native| native.bundle.as_deref()),
            Some("audiocraft")
        );
    }

    #[test]
    fn yuey_is_a_native_service_with_a_package_for_every_backend_it_offers() {
        let manifest = bundled();
        let yuey = manifest
            .services
            .iter()
            .find(|service| service.id == "yuey")
            .expect("bundled manifest should define yuey");

        assert_eq!(yuey.runtime, ServiceRuntime::Native);
        assert_eq!(yuey.port, 8007);
        let native = yuey
            .native
            .as_ref()
            .expect("yuey should carry a native block");
        assert_eq!(native.executable, "yue2-server.exe");

        let windows = native
            .platforms
            .get("windows-x64")
            .expect("yuey should publish a windows-x64 package");
        for backend in ["cuda", "vulkan"] {
            assert!(
                windows.backends.contains_key(backend),
                "yuey should offer a {backend} backend"
            );
        }
        // Measured on an RTX 5070 Laptop: with the decode step replayed as a
        // CUDA graph, CUDA is also the faster backend, and with a DAW open a
        // 170s Vulkan render stalled for over ten minutes where CUDA finished.
        // gary4juce lives in a DAW, so NVIDIA stays on CUDA.
        assert_eq!(windows.prefer, ["cuda", "vulkan"]);

        // Every runtime a backend asks for has to be one the manifest can install.
        for package in windows.backends.values() {
            for runtime in &package.requires {
                assert!(
                    manifest.native_runtimes.contains_key(runtime),
                    "{runtime} is required but not defined"
                );
            }
        }
    }
}
