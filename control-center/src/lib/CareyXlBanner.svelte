<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";

  let {
    enabled,
    serviceStatus,
    onUpdated,
  }: {
    enabled: boolean;
    serviceStatus: "stopped" | "starting" | "running" | "unhealthy" | "failed";
    onUpdated: (enabled: boolean) => void;
  } = $props();

  let saving = $state(false);
  let message: string | null = $state(null);

  async function toggleXlModels(nextEnabled: boolean) {
    saving = true;
    message = null;

    try {
      await invoke("save_app_settings", {
        settings: { careyUseXlModels: nextEnabled },
      });

      onUpdated(nextEnabled);

      if (serviceStatus === "running" || serviceStatus === "starting" || serviceStatus === "unhealthy") {
        await invoke("restart_service", { serviceId: "carey" });
        message = nextEnabled
          ? "xl models enabled. carey is restarting."
          : "regular models enabled. carey is restarting.";
      } else {
        message = nextEnabled
          ? "xl models enabled for the next carey start."
          : "regular carey models enabled for the next start.";
      }
    } catch (e: any) {
      message = "Failed: " + (typeof e === "string" ? e : e?.message || "unknown");
    } finally {
      saving = false;
    }
  }
</script>

<div class="carey-banner">
  <div class="banner-row">
    <div class="copy">
      <div class="banner-title">carey xl models</div>
      <div class="banner-subtitle">optional higher-memory mode for ace-step</div>
    </div>
    <label class="toggle">
      <input
        type="checkbox"
        checked={enabled}
        disabled={saving}
        onchange={(e) => toggleXlModels((e.currentTarget as HTMLInputElement).checked)}
      />
      <span>{enabled ? "on" : "off"}</span>
    </label>
  </div>

  <div class="note">
    xl-base, xl-sft, and xl-turbo can be tight at 12 GB. we recommend 16 GB of GPU VRAM before turning
    this on. the first xl start may also download additional carey weights before the service becomes healthy.
  </div>

  {#if message}
    <div class="msg">{message}</div>
  {/if}
</div>

<style>
  .carey-banner {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: #1b2420;
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

  .toggle input {
    accent-color: var(--accent);
  }

  .note {
    margin-top: 8px;
    font-size: 10px;
    color: var(--text-muted);
    line-height: 1.4;
  }

  .msg {
    margin-top: 6px;
    font-size: 10px;
    color: var(--text-secondary);
  }
</style>
