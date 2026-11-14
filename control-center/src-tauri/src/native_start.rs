//! Start may need a verified model choice, but downloading never starts a GPU job.
use std::future::Future;

pub async fn complete<A, AF, S, SF>(
    encoding: Option<String>, activate: A, start: S,
) -> Result<(), String>
where
    A: FnOnce(String) -> AF,
    AF: Future<Output = Result<(), String>>,
    S: FnOnce() -> SF,
    SF: Future<Output = Result<(), String>>,
{
    if let Some(encoding) = encoding { activate(encoding).await?; }
    start().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[tokio::test]
    async fn start_waits_for_successful_activation() {
        let calls = Mutex::new(Vec::new());
        let calls = &calls;
        complete(Some("Q4_K_M".into()), |encoding| async move {
            calls.lock().unwrap().push(format!("verify {encoding}"));
            tokio::task::yield_now().await;
            calls.lock().unwrap().push("selection saved".into());
            Ok(())
        }, || async { calls.lock().unwrap().push("start".into()); Ok(()) }).await.unwrap();
        assert_eq!(*calls.lock().unwrap(), ["verify Q4_K_M", "selection saved", "start"]);
    }

    #[tokio::test]
    async fn failed_verification_or_activation_never_launches_a_service() {
        for cause in ["model hash mismatch", "runtime verification failed", "cannot save selection"] {
            let result = complete(Some("F16".into()), |_| async { Err(cause.into()) },
                || async { panic!("must not start after failed activation") }).await;
            assert_eq!(result.unwrap_err(), cause);
        }
    }

    #[tokio::test]
    async fn an_existing_choice_starts_without_activation_and_preserves_start_errors() {
        let result = complete(None, |_| async { panic!("do not repeat activation") },
            || async { Err("port is busy".into()) }).await;
        assert_eq!(result.unwrap_err(), "port is busy");
    }

    #[test]
    fn automatic_setup_cleanup_excludes_environments_weights_and_warnings() {
        use crate::migration_inventory::{Item, Preview};
        let mut preview = Preview {
            active_root: String::new(), hf_hub_cache: String::new(), cleanup_candidates: vec![],
            legacy_runtime_present: false, training_base_present: false, estimated_cleanup_bytes: 0,
            preserved_paths: vec![], warnings: vec![], native_selection: None,
            cleanup_token: "review token".into(), python_still_needed: None,
        };
        assert!(preview.only_bundled_code(), "empty cleanup can finish");
        preview.cleanup_candidates.push(Item { label: "unchanged bundled source".into(), path: "source.py".into(), bytes: 1, kind: Some("code") });
        assert!(preview.only_bundled_code());
        for kind in [Some("environment"), Some("weights"), None] {
            preview.cleanup_candidates[0].kind = kind;
            assert!(!preview.only_bundled_code());
        }
        preview.cleanup_candidates[0].kind = Some("code");
        preview.legacy_runtime_present = true; assert!(!preview.only_bundled_code());
        preview.legacy_runtime_present = false;
        preview.warnings.push("source changed".into()); assert!(!preview.only_bundled_code());
        preview.warnings.clear();
        preview.python_still_needed = Some("Python-only feature".into()); assert!(!preview.only_bundled_code());
    }
}
