<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";

  type ServiceStatus = "stopped" | "starting" | "running" | "unhealthy" | "failed";
  type ModelStatus = "available" | "downloading" | "downloaded" | "failed";

  interface ModelEntry {
    id: string;
    display_name: string;
    service: string;
    size_category: string | null;
    group: string | null;
    epoch: number | null;
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

  let saving = $state(false);
  let message: string | null = $state(null);
  let scragStatus = $state<ModelStatus>("available");

  const scragVaeUrl = "https://huggingface.co/scragnog/Ace-Step-1.5-ScragVAE";
  let scragDownloaded = $derived(scragStatus === "downloaded");
  let canEnable = $derived(scragDownloaded || enabled);

  function applyScragStatus(models: ModelEntry[]) {
    const scrag = models.find((m) => m.id === "carey::scrag-vae");
    scragStatus = scrag?.status ?? "available";
  }

  async function loadScragStatus() {
    try {
      applyScragStatus(await invoke<ModelEntry[]>("get_models"));
    } catch (e) {
      console.warn("Failed to load ScragVAE model status:", e);
    }
  }

  async function openScragVaePage() {
    try {
      await invoke("open_url", { url: scragVaeUrl });
    } catch (e) {
      console.error("Failed to open ScragVAE page:", e);
    }
  }

  async function toggleScragVae(nextEnabled: boolean) {
    if (nextEnabled && !scragDownloaded) {
      message = "download ScragVAE first, then this switch will wake up.";
      return;
    }

    saving = true;
    message = null;

    try {
      await invoke("save_app_settings", {
        settings: { careyUseScragVae: nextEnabled },
      });

      onUpdated(nextEnabled);

      if (serviceStatus === "running" || serviceStatus === "starting" || serviceStatus === "unhealthy") {
        await invoke("restart_service", { serviceId: "carey" });
        message = nextEnabled
          ? "ScragVAE enabled. carey is restarting."
          : "stock VAE enabled. carey is restarting.";
      } else {
        message = nextEnabled
          ? "ScragVAE enabled for the next carey start."
          : "stock VAE enabled for the next carey start.";
      }
    } catch (e: any) {
      message = "Failed: " + (typeof e === "string" ? e : e?.message || "unknown");
    } finally {
      saving = false;
    }
  }

  onMount(() => {
    void loadScragStatus();

    const unlistenModels = listen<ModelEntry[]>("models-updated", (event) => {
      applyScragStatus(event.payload);
    });

    return () => {
      unlistenModels.then((fn) => fn());
    };
  });
</script>

<div class="carey-banner scrag-banner" class:missing={!scragDownloaded}>
  <div class="banner-row">
    <div class="copy">
      <div class="banner-title">ScragVAE decoder</div>
      <div class="banner-subtitle">optional clarity boost for ace-step</div>
    </div>
    <label class="toggle" class:disabled={!canEnable || saving}>
      <input
        type="checkbox"
        checked={enabled}
        disabled={saving || !canEnable}
        onchange={(e) => toggleScragVae((e.currentTarget as HTMLInputElement).checked)}
      />
      <span>{enabled ? "on" : "off"}</span>
    </label>
  </div>

  <div class="note">
    ScragVAE is a drop-in alternate decoder by scragnog. it often adds a little more air and detail;
    stock VAE stays one click away if a render feels strange.
    <button type="button" class="inline-link" onclick={openScragVaePage}>thank scragnog / view model</button>
  </div>

  {#if !scragDownloaded}
    <div class="warning-row">
      <span>
        download ScragVAE from carey's model list before enabling it.
        {#if scragStatus === "downloading"}
          downloading now...
        {/if}
      </span>
      <button type="button" class="mini-btn" onclick={onShowModels}>open models</button>
    </div>
  {/if}

  {#if message}
    <div class="msg">{message}</div>
  {/if}
</div>

<style>
  .carey-banner {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: #221f28;
  }

  .carey-banner.missing {
    background: #211f25;
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
