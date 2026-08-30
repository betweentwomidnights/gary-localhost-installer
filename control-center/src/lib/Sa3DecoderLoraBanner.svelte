<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";

  type ServiceStatus = "stopped" | "starting" | "running" | "unhealthy" | "failed";
  type ModelStatus = "available" | "downloading" | "downloaded" | "failed";

  interface ModelEntry {
    id: string;
    status: ModelStatus;
  }

  let {
    enabled,
    serviceStatus,
    onUpdated,
    onShowModels,
  }: {
    enabled: boolean;
    serviceStatus: ServiceStatus;
    onUpdated: (enabled: boolean) => void;
    onShowModels: () => void;
  } = $props();

  const modelId = "thepatch/same-l-decoder-lora";
  const modelUrl = `https://huggingface.co/${modelId}`;

  let saving = $state(false);
  let message: string | null = $state(null);
  let modelStatus = $state<ModelStatus>("available");

  let downloaded = $derived(modelStatus === "downloaded");
  let canEnable = $derived(downloaded || enabled);

  function applyModelStatus(models: ModelEntry[]) {
    modelStatus = models.find((model) => model.id === modelId)?.status ?? "available";
  }

  async function loadModelStatus() {
    try {
      applyModelStatus(await invoke<ModelEntry[]>("get_models"));
    } catch (e) {
      console.warn("Failed to load decoder LoRA status:", e);
    }
  }

  async function openModelPage() {
    try {
      await invoke("open_url", { url: modelUrl });
    } catch (e) {
      console.error("Failed to open decoder LoRA page:", e);
    }
  }

  async function toggleDecoderLora(nextEnabled: boolean) {
    if (nextEnabled && !downloaded) {
      message = "download the decoder fix first, then this switch will wake up.";
      return;
    }

    saving = true;
    message = null;

    try {
      await invoke("save_app_settings", {
        settings: { sa3UseDecoderLora: nextEnabled },
      });
      onUpdated(nextEnabled);

      if (serviceStatus === "running" || serviceStatus === "starting" || serviceStatus === "unhealthy") {
        await invoke("restart_service", { serviceId: "sa3" });
        message = nextEnabled
          ? "decoder squeak fix enabled. sa3 is restarting."
          : "stock decoder enabled. sa3 is restarting.";
      } else {
        message = nextEnabled
          ? "decoder squeak fix enabled for the next sa3 start."
          : "stock decoder enabled for the next sa3 start.";
      }
    } catch (e: any) {
      message = "Failed: " + (typeof e === "string" ? e : e?.message || "unknown");
    } finally {
      saving = false;
    }
  }

  onMount(() => {
    void loadModelStatus();
    const unlistenModels = listen<ModelEntry[]>("models-updated", (event) => {
      applyModelStatus(event.payload);
    });
    return () => {
      unlistenModels.then((fn) => fn());
    };
  });
</script>

<div class="sa3-banner" class:missing={!downloaded}>
  <div class="banner-row">
    <div class="copy">
      <div class="banner-title">decoder squeak fix</div>
      <div class="banner-subtitle">optional SAME-L decoder LoRA</div>
    </div>
    <label class="toggle" class:disabled={!canEnable || saving}>
      <input
        type="checkbox"
        checked={enabled}
        disabled={saving || !canEnable}
        onchange={(event) => toggleDecoderLora((event.currentTarget as HTMLInputElement).checked)}
      />
      <span>{enabled ? "on" : "off"}</span>
    </label>
  </div>

  <div class="note">
    stabilizes repeated SAME-L encode/decode passes that can build up transient squeaks.
    sa3 merges it into the decoder at model load, so generation keeps the stock decode path.
    <button type="button" class="inline-link" onclick={openModelPage}>view model</button>
  </div>

  {#if !downloaded}
    <div class="warning-row">
      <span>
        download the decoder fix from sa3's model list before enabling it.
        {#if modelStatus === "downloading"} downloading now... {/if}
      </span>
      <button type="button" class="mini-btn" onclick={onShowModels}>open models</button>
    </div>
  {/if}

  {#if message}
    <div class="msg">{message}</div>
  {/if}
</div>

<style>
  .sa3-banner {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: #1d2428;
  }

  .sa3-banner.missing {
    background: #202226;
  }

  .banner-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .copy {
    flex: 1;
    min-width: 0;
  }

  .banner-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .banner-subtitle {
    margin-top: 3px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .toggle {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-primary);
    white-space: nowrap;
  }

  .toggle.disabled {
    color: var(--text-muted);
  }

  .toggle input {
    accent-color: var(--accent);
  }

  .note {
    margin-top: 8px;
    font-size: 10px;
    color: var(--text-muted);
    line-height: 1.4;
  }

  .inline-link {
    margin-left: 6px;
    padding: 0;
    border: none;
    background: transparent;
    color: #ffffff;
    font: inherit;
    font-weight: 700;
    cursor: pointer;
  }

  .inline-link:hover {
    text-decoration: underline;
  }

  .warning-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-top: 8px;
    padding: 6px 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    color: var(--text-secondary);
    font-size: 10px;
    line-height: 1.35;
  }

  .mini-btn {
    flex: 0 0 auto;
    padding: 2px 8px;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: transparent;
    color: var(--text-primary);
    font-size: 10px;
    cursor: pointer;
  }

  .mini-btn:hover {
    background: var(--bg-hover);
  }

  .msg {
    margin-top: 6px;
    font-size: 10px;
    color: var(--text-secondary);
  }
</style>
