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

  async function toggleFlashAttention(nextEnabled: boolean) {
    saving = true;
    message = null;

    try {
      await invoke("save_app_settings", {
        settings: { melodyflowUseFlashAttn: nextEnabled },
      });

      onUpdated(nextEnabled);

      if (serviceStatus === "running" || serviceStatus === "starting" || serviceStatus === "unhealthy") {
        await invoke("restart_service", { serviceId: "melodyflow" });
        message = nextEnabled
          ? "flash attention enabled. terry is restarting."
          : "flash attention disabled. terry is restarting.";
      } else {
        message = nextEnabled
          ? "flash attention enabled for the next terry start."
          : "flash attention disabled. terry will use the standard path next start.";
      }
    } catch (e: any) {
      message = "Failed: " + (typeof e === "string" ? e : e?.message || "unknown");
    } finally {
      saving = false;
    }
  }
</script>

<div class="flash-banner">
  <div class="banner-row">
    <div class="copy">
      <div class="banner-title">melodyflow flash attention</div>
      <div class="banner-subtitle">optional speed mode for terry</div>
    </div>
    <label class="toggle">
      <input
        type="checkbox"
        checked={enabled}
        disabled={saving}
        onchange={(e) => toggleFlashAttention((e.currentTarget as HTMLInputElement).checked)}
      />
      <span>{enabled ? "on" : "off"}</span>
    </label>
  </div>

  <div class="note">
    can speed generation up, but may sound worse than the default attention path on some material.
  </div>

  {#if message}
    <div class="msg">{message}</div>
  {/if}
</div>

<style>
  .flash-banner {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: #1a2028;
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
