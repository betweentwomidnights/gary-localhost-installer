<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";

  interface Sa3LoudnessSettings {
    peakNormalizeDb: string;
    limiterCeilingDb: string;
    latentRescale: string;
    latentShift: string;
    latentTargetStd: string;
    continuationTailPad: string;
  }

  let {
    settings,
    serviceStatus,
    onUpdated,
  }: {
    settings: Sa3LoudnessSettings;
    serviceStatus: "stopped" | "starting" | "running" | "unhealthy" | "failed";
    onUpdated: (settings: Sa3LoudnessSettings) => void;
  } = $props();

  const defaults: Sa3LoudnessSettings = {
    peakNormalizeDb: "2.0",
    limiterCeilingDb: "-0.3",
    latentRescale: "1.0",
    latentShift: "0.0",
    latentTargetStd: "",
    continuationTailPad: "6",
  };

  let tab: "level" | "latent" | "continue" = $state("level");
  let draft: Sa3LoudnessSettings = $state({ ...defaults });
  let saving = $state(false);
  let message: string | null = $state(null);

  $effect(() => {
    draft = { ...settings };
  });

  function cleanDraft(): Sa3LoudnessSettings {
    return {
      peakNormalizeDb: draft.peakNormalizeDb.trim(),
      limiterCeilingDb: draft.limiterCeilingDb.trim(),
      latentRescale: draft.latentRescale.trim(),
      latentShift: draft.latentShift.trim(),
      latentTargetStd: draft.latentTargetStd.trim(),
      continuationTailPad: draft.continuationTailPad.trim(),
    };
  }

  function resetDefaults() {
    draft = { ...defaults };
    message = null;
  }

  async function saveSettings() {
    saving = true;
    message = null;

    try {
      const updated = await invoke<{ sa3Loudness: Sa3LoudnessSettings }>("save_app_settings", {
        settings: { sa3Loudness: cleanDraft() },
      });
      onUpdated(updated.sa3Loudness);

      if (serviceStatus === "running" || serviceStatus === "starting" || serviceStatus === "unhealthy") {
        await invoke("restart_service", { serviceId: "sa3" });
        message = "output defaults saved. sa3 is restarting.";
      } else {
        message = "output defaults saved for the next sa3 start.";
      }
    } catch (e: any) {
      message = "Failed: " + (typeof e === "string" ? e : e?.message || "unknown");
    } finally {
      saving = false;
    }
  }
</script>

<div class="output-panel">
  <div class="panel-head">
    <div class="copy">
      <div class="panel-title">sa3 output shaping</div>
      <div class="panel-subtitle">advanced loudness and continuation defaults</div>
    </div>
    <div class="tabs" aria-label="sa3 output shaping tabs">
      <button class:active={tab === "level"} onclick={() => tab = "level"}>level</button>
      <button class:active={tab === "latent"} onclick={() => tab = "latent"}>latent</button>
      <button class:active={tab === "continue"} onclick={() => tab = "continue"}>continue</button>
    </div>
  </div>

  {#if tab === "level"}
    <div class="field-grid">
      <label class="field">
        <span>peak normalize dB</span>
        <input bind:value={draft.peakNormalizeDb} placeholder="2.0" />
        <small>Pre-scales the decoded waveform before the limiter. Use off to disable.</small>
      </label>
      <label class="field">
        <span>limiter ceiling dB</span>
        <input bind:value={draft.limiterCeilingDb} placeholder="-0.3" />
        <small>Soft anti-clip ceiling. Keep it at or below 0, or use off.</small>
      </label>
    </div>
  {:else if tab === "latent"}
    <div class="field-grid">
      <label class="field">
        <span>latent rescale</span>
        <input bind:value={draft.latentRescale} placeholder="1.0" />
        <small>Constant multiply before decode. 1.0 leaves latents unchanged.</small>
      </label>
      <label class="field">
        <span>latent shift</span>
        <input bind:value={draft.latentShift} placeholder="0.0" />
        <small>Constant offset before decode. 0.0 leaves latents unchanged.</small>
      </label>
      <label class="field wide">
        <span>adaptive target std</span>
        <input bind:value={draft.latentTargetStd} placeholder="off" />
        <small>Optional adaptive attenuation for hot LoRAs. Empty or off disables; try 0.9.</small>
      </label>
    </div>
  {:else}
    <div class="field-grid">
      <label class="field wide">
        <span>continuation tail pad seconds</span>
        <input bind:value={draft.continuationTailPad} placeholder="6" />
        <small>For continue mode: 0 ends at the cut, 6 keeps a natural tail, 20+ leans seamless.</small>
      </label>
    </div>
  {/if}

  <div class="actions">
    <button class="accent" onclick={saveSettings} disabled={saving}>save</button>
    <button onclick={resetDefaults} disabled={saving}>defaults</button>
    {#if message}
      <span class="message">{message}</span>
    {/if}
  </div>
</div>

<style>
  .output-panel {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: #1b2024;
  }

  .panel-head {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .copy {
    flex: 1;
    min-width: 0;
  }

  .panel-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .panel-subtitle {
    margin-top: 3px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .tabs {
    display: inline-flex;
    flex-shrink: 0;
    border: 1px solid var(--border);
  }

  .tabs button {
    border: 0;
    border-right: 1px solid var(--border);
    padding: 4px 8px;
    font-size: 10px;
    background: transparent;
    color: var(--text-secondary);
  }

  .tabs button:last-child {
    border-right: 0;
  }

  .tabs button.active {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .field-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin-top: 10px;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 5px;
    min-width: 0;
  }

  .field.wide {
    grid-column: 1 / -1;
  }

  .field span {
    font-size: 10px;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
  }

  .field input {
    width: 100%;
    min-width: 0;
    padding: 5px 7px;
    border: 1px solid var(--border);
    background: var(--bg-panel);
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 11px;
    outline: none;
  }

  .field input:focus {
    border-color: var(--accent);
  }

  .field input::placeholder {
    color: var(--text-muted);
  }

  .field small {
    color: var(--text-muted);
    font-size: 10px;
    line-height: 1.35;
  }

  .actions {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
  }

  .actions button {
    padding: 4px 10px;
    font-size: 10px;
  }

  .message {
    min-width: 0;
    color: var(--text-secondary);
    font-size: 10px;
  }
</style>
