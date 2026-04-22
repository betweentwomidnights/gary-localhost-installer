<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { open as openDialog } from "@tauri-apps/plugin-dialog";

  interface CareyLoraEntry {
    name: string;
    path: string;
    captionsPath: string | null;
    resolvedCaptionsPath: string | null;
    captionCount: number;
    scale: number;
    backends: string[];
    modelFamily: "standard" | "xl";
    checkpointExists: boolean;
    registered: boolean;
  }

  interface CareyLoraState {
    entries: CareyLoraEntry[];
    pools: Record<string, number>;
    catalogPath: string;
    registryPath: string;
    captionsPath: string;
  }

  interface CareyCaptionsBuildResult {
    state: CareyLoraState;
    output: string;
  }

  let {
    open,
    serviceStatus,
    serviceEnvExists,
    careyXlEnabled,
    onClose,
  }: {
    open: boolean;
    serviceStatus: "stopped" | "starting" | "running" | "unhealthy" | "failed";
    serviceEnvExists: boolean;
    careyXlEnabled: boolean;
    onClose: () => void;
  } = $props();

  let loraState: CareyLoraState | null = $state(null);
  let loading = $state(false);
  let saving = $state(false);
  let building = $state(false);
  let message: string | null = $state(null);
  let error: string | null = $state(null);
  let buildOutput: string | null = $state(null);

  let formName = $state("");
  let checkpointPath = $state("");
  let captionsPath = $state("");
  let modelFamily = $state<"standard" | "xl">("standard");

  function describeError(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }

  function basename(path: string): string {
    const normalized = path.replace(/\\/g, "/").replace(/\/+$/, "");
    const parts = normalized.split("/");
    return parts[parts.length - 1] || "";
  }

  function suggestName(path: string): string {
    const raw = basename(path).toLowerCase();
    return raw
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 64);
  }

  function clearForm() {
    formName = "";
    checkpointPath = "";
    captionsPath = "";
    modelFamily = careyXlEnabled ? "xl" : "standard";
  }

  async function loadState() {
    loading = true;
    error = null;
    try {
      loraState = await invoke<CareyLoraState>("get_carey_lora_state");
    } catch (e) {
      error = describeError(e);
    } finally {
      loading = false;
    }
  }

  async function pickCheckpointFolder() {
    const selected = await openDialog({ directory: true, multiple: false });
    if (typeof selected !== "string") return;
    checkpointPath = selected;
    if (!formName.trim()) {
      formName = suggestName(selected);
    }
    if (selected.toLowerCase().includes("xl")) {
      modelFamily = "xl";
    }
  }

  async function pickCaptionsFolder() {
    const selected = await openDialog({ directory: true, multiple: false });
    if (typeof selected !== "string") return;
    captionsPath = selected;
    if (!formName.trim()) {
      formName = suggestName(selected);
    }
  }

  async function saveLora() {
    saving = true;
    error = null;
    message = null;
    buildOutput = null;
    try {
      loraState = await invoke<CareyLoraState>("upsert_carey_lora", {
        name: formName,
        checkpointPath,
        captionsPath: captionsPath.trim() ? captionsPath : null,
        modelFamily,
      });
      message = `saved ${formName.trim().toLowerCase()}`;
      clearForm();
    } catch (e) {
      error = describeError(e);
    } finally {
      saving = false;
    }
  }

  async function removeLora(name: string) {
    error = null;
    message = null;
    buildOutput = null;
    try {
      loraState = await invoke<CareyLoraState>("remove_carey_lora", { name });
      message = `removed ${name}`;
    } catch (e) {
      error = describeError(e);
    }
  }

  async function buildCaptions() {
    building = true;
    error = null;
    message = null;
    buildOutput = null;
    try {
      const result = await invoke<CareyCaptionsBuildResult>("build_carey_lora_captions");
      loraState = result.state;
      buildOutput = result.output;
      message = "captions.json updated";
    } catch (e) {
      error = describeError(e);
    } finally {
      building = false;
    }
  }

  async function openExamplesRepo() {
    await invoke("open_url", { url: "https://github.com/betweentwomidnights/gary-lora-examples" });
  }

  let canBuild = $derived(
    open &&
    serviceEnvExists &&
    serviceStatus === "running" &&
    !building
  );

  let poolSummary: [string, number][] = $derived.by(() => {
    if (!loraState) return [];
    const entries = Object.entries(loraState.pools) as [string, number][];
    entries.sort((a, b) => a[0].localeCompare(b[0]));
    return entries;
  });

  $effect(() => {
    if (open) {
      if (!formName.trim() && !checkpointPath.trim() && !captionsPath.trim()) {
        modelFamily = careyXlEnabled ? "xl" : "standard";
      }
      void loadState();
    }
  });
</script>

{#if open}
  <div class="overlay">
    <button type="button" class="backdrop" aria-label="close carey lora manager" onclick={onClose}></button>
    <div
      class="modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="carey-lora-title"
      tabindex="-1"
    >
      <div class="eyebrow">carey lora manager</div>
      <div class="title" id="carey-lora-title">add local LoRAs and caption pools</div>
      <div class="body">
        pick the LoRA checkpoint folder that Carey should load. if your training sidecars live somewhere else,
        add that folder too and gary4local will use it only for captions generation.
      </div>

      <div class="note">
        this keeps the runtime registry compatible with the remote backend while still supporting the separate
        staging-folder workflow you used for training.
      </div>

      <div class="note">
        the model family tag matters: gary4juce only sees LoRAs that match the current Carey mode. when xl is on,
        xl LoRAs are exposed; when xl is off, standard LoRAs are exposed.
      </div>

      {#if !serviceEnvExists}
        <div class="warning">build Carey first. you can still save LoRA entries now, but captions generation is unavailable.</div>
      {:else if serviceStatus !== "running"}
        <div class="warning">Carey is not running. you can save LoRA entries now; start Carey to build captions and hot-reload the registry.</div>
      {/if}

      <div class="section-label">new entry</div>
      <div class="form-grid">
        <label class="field">
          <span>LoRA name</span>
          <input type="text" bind:value={formName} placeholder="mayer" />
        </label>

        <label class="field">
          <span>model family</span>
          <select bind:value={modelFamily}>
            <option value="standard">standard</option>
            <option value="xl">xl</option>
          </select>
        </label>

        <label class="field wide">
          <span>checkpoint folder</span>
          <div class="path-row">
            <input type="text" bind:value={checkpointPath} placeholder="C:\\path\\to\\adapter folder" />
            <button type="button" onclick={pickCheckpointFolder}>pick folder</button>
          </div>
        </label>

        <label class="field wide">
          <span>captions/source folder (optional)</span>
          <div class="path-row">
            <input type="text" bind:value={captionsPath} placeholder="C:\\path\\to\\sidecars folder" />
            <button type="button" onclick={pickCaptionsFolder}>pick folder</button>
          </div>
        </label>
      </div>

      <div class="note">
        if your `.txt` sidecars live beside the adapter already, leave the captions/source folder blank.
      </div>

      <div class="actions top-actions">
        <button
          class="accent"
          onclick={saveLora}
          disabled={saving || !formName.trim() || !checkpointPath.trim()}
        >
          {saving ? "saving..." : "save LoRA"}
        </button>
        <button onclick={() => void loadState()} disabled={loading || saving || building}>refresh</button>
        <button onclick={buildCaptions} disabled={!canBuild}>
          {building ? "building..." : "build captions"}
        </button>
        <button onclick={openExamplesRepo}>open examples repo</button>
        <button onclick={onClose}>close</button>
      </div>

      {#if message}
        <div class="success-note">{message}</div>
      {/if}
      {#if error}
        <div class="error-note">{error}</div>
      {/if}
      {#if buildOutput}
        <pre class="output">{buildOutput}</pre>
      {/if}

      <div class="section-label">registered LoRAs</div>
      {#if loading && !loraState}
        <div class="empty">loading...</div>
      {:else if loraState && loraState.entries.length > 0}
        <div class="entry-list">
          {#each loraState.entries as entry}
            <div class="entry-card">
              <div class="entry-top">
                <div>
                  <div class="entry-name">{entry.name}</div>
                  <div class="entry-meta">{entry.modelFamily} · backends: {entry.backends.join(", ")} · scale {entry.scale}</div>
                </div>
                <button class="danger" onclick={() => removeLora(entry.name)}>remove</button>
              </div>

              <div class="entry-path">checkpoint: {entry.path}</div>
              {#if entry.captionsPath}
                <div class="entry-path">captions source: {entry.captionsPath}</div>
              {:else if entry.resolvedCaptionsPath}
                <div class="entry-path">captions source: using checkpoint folder sidecars</div>
              {:else}
                <div class="entry-path">captions source: not set</div>
              {/if}

              <div class="entry-meta">
                {entry.captionCount} caption sidecars detected
                {#if !entry.registered}
                  · checkpoint folder missing or invalid
                {/if}
              </div>

              {#if entry.captionCount === 0}
                <div class="note">
                  no sidecar txts found. the dice button will fall back to the plugin's default pool.
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {:else}
        <div class="empty">no LoRAs added yet.</div>
      {/if}

      {#if poolSummary.length > 0}
        <div class="section-label">captions pools</div>
        <div class="pool-list">
          {#each poolSummary as [name, count]}
            <div class="pool-row">
              <span>{name}</span>
              <span>{count}</span>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .overlay {
    position: fixed;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    z-index: 70;
  }

  .backdrop {
    position: absolute;
    inset: 0;
    border: none;
    background: rgba(0, 0, 0, 0.72);
    padding: 0;
  }

  .modal {
    position: relative;
    z-index: 1;
    width: min(760px, 100%);
    max-height: min(88vh, 980px);
    overflow: auto;
    border: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(34, 34, 34, 0.98), rgba(18, 18, 18, 0.98));
    box-shadow: 0 22px 64px rgba(0, 0, 0, 0.5);
    padding: 20px;
  }

  .eyebrow {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-secondary);
  }

  .title {
    margin-top: 8px;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .body {
    margin-top: 12px;
    font-size: 13px;
    color: var(--text-primary);
    line-height: 1.55;
  }

  .note,
  .warning,
  .success-note,
  .error-note,
  .entry-meta,
  .entry-path {
    margin-top: 8px;
    font-size: 11px;
    line-height: 1.45;
    word-break: break-word;
  }

  .note,
  .entry-meta,
  .entry-path {
    color: var(--text-secondary);
  }

  .warning {
    color: #ffcb8f;
  }

  .success-note {
    color: #9bd8aa;
  }

  .error-note {
    color: #ff8f8f;
  }

  .section-label {
    margin-top: 18px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary);
    font-weight: 600;
  }

  .form-grid {
    display: grid;
    gap: 12px;
    margin-top: 12px;
  }

  .field {
    display: grid;
    gap: 6px;
  }

  .field span {
    font-size: 11px;
    color: var(--text-secondary);
  }

  .field.wide {
    grid-column: 1 / -1;
  }

  .path-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 8px;
  }

  .actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 14px;
  }

  .top-actions {
    margin-top: 16px;
  }

  .actions .accent {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }

  .actions .accent:hover:not(:disabled) {
    background: var(--accent);
    border-color: var(--accent-hover);
  }

  select {
    width: 100%;
    font-family: var(--font-sans);
    border: 1px solid var(--border);
    background: var(--bg-panel);
    color: var(--text-primary);
    padding: 8px 10px;
    font-size: 12px;
  }

  .output {
    margin-top: 12px;
    padding: 10px;
    border: 1px solid var(--border);
    background: rgba(0, 0, 0, 0.24);
    color: var(--text-primary);
    font-size: 11px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .entry-list {
    display: grid;
    gap: 12px;
    margin-top: 12px;
  }

  .entry-card {
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.02);
    padding: 12px;
  }

  .entry-top {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: flex-start;
  }

  .entry-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .danger {
    border-color: #b24a4a;
    color: #ffb1b1;
  }

  .danger:hover:not(:disabled) {
    background: #7a2626;
    color: white;
  }

  .pool-list {
    margin-top: 12px;
    border: 1px solid var(--border);
  }

  .pool-row {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    padding: 8px 10px;
    font-size: 12px;
    color: var(--text-primary);
  }

  .pool-row + .pool-row {
    border-top: 1px solid var(--border);
  }

  .empty {
    margin-top: 12px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  @media (max-width: 700px) {
    .overlay {
      padding: 12px;
    }

    .modal {
      padding: 16px;
      max-height: 94vh;
    }

    .path-row {
      grid-template-columns: 1fr;
    }
  }
</style>
