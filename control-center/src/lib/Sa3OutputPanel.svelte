<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";

  interface Sa3LoudnessSettings {
    peakNormalizeDb: string;
    limiterCeilingDb: string;
    latentRescale: string;
    latentShift: string;
    latentTargetStd: string;
    continuationTailPad: string;
    continuationSpliceSource: boolean;
    continuationSpliceXfade: string;
    continuationSpliceGainMatch: boolean;
    continuationMaskOverlap: string;
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
    continuationSpliceSource: true,
    continuationSpliceXfade: "0.03",
    continuationSpliceGainMatch: true,
    continuationMaskOverlap: "0.2",
  };

  let tab: "level" | "latent" | "tail" | "splice" = $state("level");
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
      continuationSpliceSource: draft.continuationSpliceSource,
      continuationSpliceXfade: draft.continuationSpliceXfade.trim(),
      continuationSpliceGainMatch: draft.continuationSpliceGainMatch,
      continuationMaskOverlap: draft.continuationMaskOverlap.trim(),
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
      const cleaned = cleanDraft();
      const spliceXfade = Number(cleaned.continuationSpliceXfade);
      if (!cleaned.continuationSpliceXfade || !Number.isFinite(spliceXfade) || spliceXfade < 0 || spliceXfade > 1) {
        message = "Crossfade must be a number from 0 to 1 second.";
        return;
      }
      const maskOverlap = Number(cleaned.continuationMaskOverlap);
      if (!cleaned.continuationMaskOverlap || !Number.isFinite(maskOverlap) || maskOverlap < 0) {
        message = "Mask overlap must be zero or a positive number of seconds.";
        return;
      }
      const updated = await invoke<{ sa3Loudness: Sa3LoudnessSettings }>("save_app_settings", {
        settings: { sa3Loudness: cleaned },
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
      <div class="panel-subtitle">advanced loudness and generation defaults</div>
    </div>
    <div class="tabs" aria-label="sa3 output shaping tabs">
      <button class:active={tab === "level"} onclick={() => tab = "level"}>level</button>
      <button class:active={tab === "latent"} onclick={() => tab = "latent"}>latent</button>
      <button class:active={tab === "tail"} onclick={() => tab = "tail"}>tail</button>
      <button class:active={tab === "splice"} onclick={() => tab = "splice"}>splice</button>
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
  {:else if tab === "tail"}
    <div class="field-grid">
      <label class="field wide">
        <span>tail pad seconds</span>
        <input bind:value={draft.continuationTailPad} placeholder="6" />
        <small>For generate and continue: adds ending headroom, then trims to the requested length.</small>
      </label>
    </div>
  {:else}
    <div class="field-grid">
      <label class="check-field">
        <input type="checkbox" bind:checked={draft.continuationSpliceSource} />
        <span>
          <strong>restore original source</strong>
          <small>Restore the source waveform over the kept continuation head before loudness shaping.</small>
        </span>
      </label>
      <label class="check-field">
        <input type="checkbox" bind:checked={draft.continuationSpliceGainMatch} />
        <span>
          <strong>RMS gain match</strong>
          <small>Match source level to the model re-render before blending. Recommended.</small>
        </span>
      </label>
      <label class="field wide">
        <span>crossfade seconds</span>
        <input bind:value={draft.continuationSpliceXfade} placeholder="0.03" />
        <small>Equal-power blend at the seam. 0.03 is recommended; very long fades can sound phasey.</small>
      </label>
      <label class="field wide">
        <span>mask overlap seconds</span>
        <input bind:value={draft.continuationMaskOverlap} placeholder="0.2" />
        <small>Regenerates the final part of the source before continuing. 0.2 matches the remote backend; 0 disables it.</small>
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

  .check-field {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    min-width: 0;
    color: var(--text-secondary);
    font-size: 10px;
  }

  .check-field input {
    margin-top: 2px;
    accent-color: var(--accent);
  }

  .check-field span {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .check-field strong {
    color: var(--text-secondary);
    font-size: 10px;
    text-transform: uppercase;
  }

  .check-field small {
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
