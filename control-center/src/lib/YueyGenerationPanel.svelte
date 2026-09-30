<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";

  export interface YueyGenerationSettings {
    instrumentalMethod: "transfer" | "rest";
    instrumentalAdapter: boolean;
    naturalMaxSeconds: number;
  }

  let {
    settings,
    serviceStatus,
    onUpdated,
  }: {
    settings: YueyGenerationSettings;
    serviceStatus: "stopped" | "starting" | "running" | "unhealthy" | "failed";
    onUpdated: (settings: YueyGenerationSettings) => void;
  } = $props();

  // The server's own defaults. The shared backend holds natural length tighter.
  const defaults: YueyGenerationSettings = {
    instrumentalMethod: "transfer",
    instrumentalAdapter: true,
    naturalMaxSeconds: 180,
  };
  const minSeconds = 30;
  const maxSeconds = 600;

  let method: "transfer" | "rest" = $state(defaults.instrumentalMethod);
  let adapter = $state(defaults.instrumentalAdapter);
  let seconds = $state(String(defaults.naturalMaxSeconds));
  let saving = $state(false);
  let message: string | null = $state(null);

  $effect(() => {
    method = settings.instrumentalMethod;
    adapter = settings.instrumentalAdapter;
    seconds = String(settings.naturalMaxSeconds);
  });

  function resetDefaults() {
    method = defaults.instrumentalMethod;
    adapter = defaults.instrumentalAdapter;
    seconds = String(defaults.naturalMaxSeconds);
    message = null;
  }

  async function saveSettings() {
    const naturalMaxSeconds = Number(seconds.trim());
    if (!Number.isInteger(naturalMaxSeconds) || naturalMaxSeconds < minSeconds || naturalMaxSeconds > maxSeconds) {
      message = `natural length must be a whole number of seconds from ${minSeconds} to ${maxSeconds}.`;
      return;
    }
    saving = true;
    message = null;
    try {
      const updated = await invoke<{ yueyGeneration: YueyGenerationSettings }>("save_app_settings", {
        settings: {
          yueyGeneration: { instrumentalMethod: method, instrumentalAdapter: adapter, naturalMaxSeconds },
        },
      });
      onUpdated(updated.yueyGeneration);
      // The server reads these at launch.
      if (serviceStatus === "running" || serviceStatus === "starting" || serviceStatus === "unhealthy") {
        await invoke("restart_service", { serviceId: "yuey" });
        message = "saved. yuey is restarting.";
      } else {
        message = "saved for the next yuey start.";
      }
    } catch (e: any) {
      message = "Failed: " + (typeof e === "string" ? e : e?.message || "unknown");
    } finally {
      saving = false;
    }
  }
</script>

<div class="generation-panel">
  <div class="panel-title">yuey generation</div>
  <div class="panel-subtitle">defaults for every client; a request that sets one itself still wins</div>

  <div class="field-grid">
    <label
      class="field"
      title="official YuE: the melody moves to the instrument lane, so an instrument plays it. our original: the melody is taken out and only the backing part is left, which is usually sparser. both only apply with instrumental on."
    >
      <span>instrumental method</span>
      <select bind:value={method}>
        <option value="transfer">official YuE</option>
        <option value="rest">our original</option>
      </select>
      <small>
        official YuE moves the sung or planned melody onto an instrument. our
        original takes it out and keeps only the backing part.
      </small>
    </label>
    <label class="field">
      <span>natural length, seconds</span>
      <input bind:value={seconds} inputmode="numeric" placeholder="180" />
      <small>
        The longest song yuey picks for itself when no bar count is set,
        {minSeconds} to {maxSeconds}. A remix keeps its source's length.
      </small>
    </label>
    <label class="check-field wide">
      <input type="checkbox" bind:checked={adapter} />
      <span>
        <strong>instrumental adapter</strong>
        <small>
          Adds the instrumental LoRA, trained for instrumental renders, to
          instrumental jobs when it is downloaded. Turn it off to compare.
        </small>
      </span>
    </label>
  </div>

  <div class="actions">
    <button class="accent" onclick={saveSettings} disabled={saving}>save</button>
    <button onclick={resetDefaults} disabled={saving}>defaults</button>
    {#if message}
      <span class="message">{message}</span>
    {/if}
  </div>
</div>

<style>
  .generation-panel {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: #1b2024;
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

  .wide {
    grid-column: 1 / -1;
  }

  .field span {
    font-size: 10px;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
  }

  .field input,
  .field select {
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

  .field input:focus,
  .field select:focus {
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
