<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { open as openDialog } from "@tauri-apps/plugin-dialog";

  interface Sa3LoraEntry {
    name: string;
    path: string;
    promptsPath: string | null;
    resolvedPromptsPath: string | null;
    promptFilePath: string;
    promptFileExists: boolean;
    promptCount: number;
    captionCount: number;
    strength: number;
    checkpointExists: boolean;
    registered: boolean;
  }

  interface Sa3LoraState {
    entries: Sa3LoraEntry[];
    pools: Record<string, number>;
    catalogPath: string;
    registryPath: string;
    promptsDir: string;
  }

  interface Sa3PromptsBuildResult {
    state: Sa3LoraState;
    output: string;
  }

  let {
    open,
    serviceStatus,
    serviceEnvExists,
    onClose,
  }: {
    open: boolean;
    serviceStatus: "stopped" | "starting" | "running" | "unhealthy" | "failed";
    serviceEnvExists: boolean;
    onClose: () => void;
  } = $props();

  let loraState: Sa3LoraState | null = $state(null);
  let loading = $state(false);
  let saving = $state(false);
  let building = $state(false);
  let message: string | null = $state(null);
  let error: string | null = $state(null);
  let buildOutput: string | null = $state(null);

  let formName = $state("");
  let checkpointPath = $state("");
  let promptsPath = $state("");

  function describeError(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }

  function basename(path: string): string {
    const normalized = path.replace(/\\/g, "/").replace(/\/+$/, "");
    const parts = normalized.split("/");
    return parts[parts.length - 1] || "";
  }

  function suggestName(path: string): string {
    return basename(path)
      .replace(/\.(ckpt|safetensors)$/i, "")
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 64);
  }

  function clearForm() {
    formName = "";
    checkpointPath = "";
    promptsPath = "";
  }

  async function loadState() {
    loading = true;
    error = null;
    try {
      loraState = await invoke<Sa3LoraState>("get_sa3_lora_state");
    } catch (e) {
      error = describeError(e);
    } finally {
      loading = false;
    }
  }

  async function pickCheckpointFile() {
    const selected = await openDialog({
      directory: false,
      multiple: false,
      filters: [{ name: "SA3 LoRA", extensions: ["ckpt", "safetensors"] }],
    });
    if (typeof selected !== "string") return;
    checkpointPath = selected;
    if (!formName.trim()) {
      formName = suggestName(selected);
    }
  }

  async function pickPromptsFolder() {
    const selected = await openDialog({ directory: true, multiple: false });
    if (typeof selected !== "string") return;
    promptsPath = selected;
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
      loraState = await invoke<Sa3LoraState>("upsert_sa3_lora", {
        name: formName,
        checkpointPath,
        promptsPath: promptsPath.trim() ? promptsPath : null,
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
      loraState = await invoke<Sa3LoraState>("remove_sa3_lora", { name });
      message = `removed ${name}`;
    } catch (e) {
      error = describeError(e);
    }
  }

  async function buildPrompts() {
    building = true;
    error = null;
    message = null;
    buildOutput = null;
    try {
      const result = await invoke<Sa3PromptsBuildResult>("build_sa3_lora_prompts");
      loraState = result.state;
      buildOutput = result.output;
      message = "prompt JSONs updated";
    } catch (e) {
      error = describeError(e);
    } finally {
      building = false;
    }
  }

  let canBuild = $derived(open && serviceEnvExists && !building);

  let poolSummary: [string, number][] = $derived.by(() => {
    if (!loraState) return [];
    const entries = Object.entries(loraState.pools) as [string, number][];
    entries.sort((a, b) => a[0].localeCompare(b[0]));
    return entries;
  });

  $effect(() => {
    if (open) {
      void loadState();
    }
  });
</script>

{#if open}
  <div class="overlay">
    <button type="button" class="backdrop" aria-label="close sa3 lora manager" onclick={onClose}></button>
    <div
      class="modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="sa3-lora-title"
      tabindex="-1"
    >
      <div class="eyebrow">sa3 lora manager</div>
      <div class="title" id="sa3-lora-title">add local LoRAs and prompt pools</div>
      <div class="body">
        pick the SA3 LoRA checkpoint file. if you still have the training dataset folder with txt sidecars,
        add that folder too and gary4local can build a prompt dice pool for the plugin.
      </div>

      <div class="note">
        SA3 loads LoRAs into the model at startup. if the model is already resident, restart SA3 before testing a newly added LoRA.
      </div>

      {#if !serviceEnvExists}
        <div class="warning">build SA3 first. you can still save LoRA entries now, but prompt generation is unavailable.</div>
      {:else if serviceStatus !== "running"}
        <div class="warning">SA3 is not running. you can save LoRA entries now; start SA3 before generation testing.</div>
      {/if}

      <div class="section-label">new entry</div>
      <div class="form-grid">
        <label class="field">
          <span>LoRA name</span>
          <input type="text" bind:value={formName} placeholder="kev" />
        </label>

        <label class="field wide">
          <span>checkpoint file</span>
          <div class="path-row">
            <input type="text" bind:value={checkpointPath} placeholder="C:\\path\\to\\your.ckpt" />
            <button type="button" onclick={pickCheckpointFile}>pick file</button>
          </div>
        </label>

        <label class="field wide">
          <span>dataset/prompts source folder (optional)</span>
          <div class="path-row">
            <input type="text" bind:value={promptsPath} placeholder="C:\\path\\to\\training dataset" />
            <button type="button" onclick={pickPromptsFolder}>pick folder</button>
          </div>
        </label>
      </div>

      <div class="note">
        the prompt builder reads each txt sidecar, strips the bpm/key tail, and writes one JSON per LoRA.
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
        <button onclick={buildPrompts} disabled={!canBuild}>
          {building ? "building..." : "build prompts"}
        </button>
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
                  <div class="entry-meta">strength {entry.strength}</div>
                </div>
                <button class="danger" onclick={() => removeLora(entry.name)}>remove</button>
              </div>

              <div class="entry-path">checkpoint: {entry.path}</div>
              {#if entry.promptsPath}
                <div class="entry-path">prompt source: {entry.promptsPath}</div>
              {:else if entry.resolvedPromptsPath}
                <div class="entry-path">prompt source: using checkpoint folder sidecars</div>
              {:else}
                <div class="entry-path">prompt source: not set</div>
              {/if}
              <div class="entry-path">prompt JSON: {entry.promptFilePath}</div>

              <div class="entry-meta">
                {entry.captionCount} txt sidecars detected, {entry.promptCount} prompts built
                {#if !entry.registered}
                  - checkpoint file missing or invalid
                {/if}
              </div>

              {#if entry.captionCount === 0}
                <div class="note">
                  no txt sidecars found. the dice button will use defaults until a prompt JSON exists.
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {:else}
        <div class="empty">no LoRAs added yet.</div>
      {/if}

      {#if poolSummary.length > 0}
        <div class="section-label">prompt pools</div>
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
