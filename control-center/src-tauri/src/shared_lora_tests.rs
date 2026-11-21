//! Exercise the actual desktop readers and writers against headless publications.
use super::*;
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};
use gary4local_core::{
    headless::Config, lora_library, sa3_adapter::AdapterRegistry, service::Service,
};

struct Fixture(PathBuf);
impl Fixture {
    fn new(service: Service) -> (Self, Config) {
        let root = std::env::temp_dir().join(format!(
            "gary-shared-{}-{}",
            service.id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let config: Config = serde_json::from_value(json!({
            "service": service, "runtime_root": root.join("host"), "library_root": root.join("library"),
            "device":"Vulkan0", "public_port":8006, "model":"medium", "encoding":"f16"
        })).unwrap();
        let installed = PathBuf::from(config.summary()["runtime_directory"].as_str().unwrap());
        std::fs::create_dir_all(&installed).unwrap();
        std::fs::write(
            config.runtime_root.join(".gary4local-headless"),
            "gary4local-server v1\n",
        )
        .unwrap();
        std::fs::write(installed.join("BUILD-INFO.json"), json!({
            "service":service.package(), "platform":gary4local_core::runtime::current_platform(),
            "dirty":false,"tests_run":true,"commit":"fixture","ggml_commit":"fixture"
        }).to_string()).unwrap();
        (Self(root), config)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn save(path: &Path, value: serde_json::Value) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, value.to_string()).unwrap();
}

#[tokio::test]
async fn shared_sa3_checkpoints_prompts_and_strength_work_in_both_interfaces() {
    let _serial = sa3_loras::REGISTRY_TEST.lock().unwrap();
    let (_fixture, config) = Fixture::new(Service::Sa3);
    let root = config.library_root.as_ref().unwrap();
    let job = root.join("sa3/training/jobs/native-fixture");
    let run = job.join("run");
    let dataset = job.join("dataset");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::create_dir_all(&dataset).unwrap();
    std::fs::write(
        dataset.join("track.txt"),
        "Dusty drum break, 95 bpm, A minor",
    )
    .unwrap();
    save(
        &job.join("native-options.json"),
        json!({
            "name":"shared", "dataset":dataset, "fixedPrompt":"", "steps":1000,
            "rank":64,"batchSize":1,"checkpointEvery":500,"duration":30.0,
            "learningRate":0.0001,"targetLatentRms":0.0,"layerScope":"transformer-core","encoding":"F16","resume":null
        }),
    );
    let earlier = run.join("adapter-step-500.gguf");
    let final_adapter = run.join("adapter-final.gguf");
    let bytes = sa3_gguf::fixture("sa3-lora", "dit");
    std::fs::write(&earlier, &bytes).unwrap();
    std::fs::write(run.join("trainer-state-step-500.gguf"), b"state").unwrap();
    let mut final_bytes = bytes.clone();
    *final_bytes.last_mut().unwrap() ^= 1;
    std::fs::write(&final_adapter, &final_bytes).unwrap();
    save(
        &job.join("native-progress.json"),
        json!({"schema_version":1,"status":"completed","phase":"completed",
        "message":"","error":"","step":1000,"epoch":2,"max_steps":1000,"final_adapter":final_adapter,"adapter_checkpoint":""}),
    );

    lora_library::publish(&config, "shared", &final_adapter, "", &[], Some(&dataset)).unwrap();
    let state = sa3_loras::state(root).unwrap();
    assert_eq!(state.entries.len(), 1);
    assert_eq!(state.entries[0].training_checkpoints.len(), 1);
    assert_eq!(state.entries[0].prompts_path.as_deref(), dataset.to_str());
    let pool = root.join("sa3/prompts/shared.json");
    let prompts = std::fs::read(&pool).unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&prompts).unwrap()["dice"]["instrumental"],
        json!(["Dusty drum break"])
    );

    sa3_loras::set_strength(root, "shared", 0.75).await.unwrap();
    sa3_training::select_checkpoint(root, "shared", &earlier)
        .await
        .unwrap();
    let headless = lora_library::Registry(root.clone()).entries().unwrap();
    assert_eq!(headless[0].strength, 0.75);
    assert_eq!(
        std::fs::read(headless[0].native_path.as_ref().unwrap()).unwrap(),
        bytes
    );
    lora_library::select_checkpoint(&config, "shared", &final_adapter).unwrap();
    let desktop = sa3_loras::state(root).unwrap();
    assert_eq!(desktop.entries[0].strength, 0.75);
    assert_eq!(
        std::fs::read(desktop.entries[0].native_path.as_ref().unwrap()).unwrap(),
        final_bytes
    );
    assert_eq!(std::fs::read(pool).unwrap(), prompts);

    // A desktop import is immediately visible to the headless server too.
    sa3_loras::register_trained(root, "desktop-import", &earlier)
        .await
        .unwrap();
    assert!(lora_library::Registry(root.clone())
        .entries()
        .unwrap()
        .iter()
        .any(|e| e.name == "desktop-import"));
}

fn peft(path: &Path, weights: &[u8]) -> PathBuf {
    let relative = ".checkpoint-generations/generation-1";
    let generation = path.join(relative);
    std::fs::create_dir_all(&generation).unwrap();
    // A text pointer exercises the portable trainer format on both platforms.
    std::fs::write(path.join("checkpoint.current"), relative).unwrap();
    save(
        &generation.join("adapter_config.json"),
        json!({"peft_type":"LORA","r":64}),
    );
    std::fs::write(generation.join("adapter_model.safetensors"), weights).unwrap();
    generation
}

#[test]
fn shared_carey_family_checkpoints_and_dice_prompts_work_in_both_interfaces() {
    let (_fixture, config) = Fixture::new(Service::Carey);
    let root = config.library_root.as_ref().unwrap();
    let job = root.join("carey/training/jobs/native-fixture");
    let run = job.join("run");
    let earlier = peft(&run.join("checkpoint-epoch-1"), b"earlier");
    let final_adapter = peft(&run, b"final");
    save(&job.join("native-options.json"), json!({"name":"shared"}));
    save(
        &job.join("native-progress.json"),
        json!({"schema_version":1,"status":"completed","phase":"completed",
        "message":"","error":"","step":1000,"epoch":2,"max_steps":1000,"final_adapter":run,"adapter_checkpoint":""}),
    );
    let captions = vec!["A quiet synth melody".to_string()];
    lora_library::publish(&config, "shared", &run, "xl", &captions, None).unwrap();
    let path = carey_lora_catalog_path_for(root);
    let mut desktop = read_carey_lora_catalog_from(&path).unwrap();
    assert_eq!(desktop["shared"].model_family, "xl");
    assert_eq!(desktop["shared"].training_checkpoints.len(), 2);
    assert_eq!(
        desktop["shared"].selected_training_checkpoint.as_deref(),
        Some("final")
    );

    // These are the same writers used by the desktop checkpoint command.
    carey_training::publish_adapter(root, "shared", &earlier).unwrap();
    let entry = desktop.get_mut("shared").unwrap();
    entry.path = earlier.to_string_lossy().into();
    entry.selected_training_checkpoint = Some("epoch-1".into());
    entry.scale = 0.65;
    save_carey_lora_catalog_to(&path, &desktop).unwrap();
    carey_training::write_adapter_info(root).unwrap();
    lora_library::select_checkpoint(
        &config,
        "shared",
        &final_adapter.join("adapter_model.safetensors"),
    )
    .unwrap();
    let desktop = read_carey_lora_catalog_from(&path).unwrap();
    assert_eq!(
        desktop["shared"].selected_training_checkpoint.as_deref(),
        Some("final")
    );
    assert_eq!(desktop["shared"].scale, 0.65);
    assert_eq!(desktop["shared"].model_family, "xl");
    let published = root.join("carey/adapters/shared");
    assert_eq!(
        std::fs::read(published.join("adapter_model.safetensors")).unwrap(),
        b"final"
    );
    let info: serde_json::Value =
        serde_json::from_slice(&std::fs::read(published.join("adapter_info.json")).unwrap())
            .unwrap();
    assert_eq!(info, json!({"model_family":"xl","captions":captions}));
    carey_training::publish_catalog_adapters(root).unwrap();
    assert_eq!(
        std::fs::read(published.join("adapter_model.safetensors")).unwrap(),
        b"final"
    );
}
