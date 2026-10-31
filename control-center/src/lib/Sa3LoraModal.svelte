<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { open as openDialog } from "@tauri-apps/plugin-dialog";
  import { rememberedDialogDirectory, rememberDialogSelection } from "./dialogMemory";

  // One list of SA3 LoRAs. Two registries sit behind it: the legacy catalog
  // (Python-era entries and Python training runs) and the native catalog
  // (C++ imports and C++ training runs, plus a prepared copy of every legacy
  // entry). Each LoRA shows once, and each action goes to whichever registry
  // owns it.

  interface Sa3TrainingCheckpoint {
    step: number;
    epoch: number;
    path: string;
  }

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
    trainingJobId: string | null;
    trainingCheckpoints: Sa3TrainingCheckpoint[];
    selectedTrainingStep: number | null;
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

  interface NativeLora {
    name: string;
    nativeOnly: boolean;
    strength: number;
    sourcePath: string;
    nativePath: string | null;
    converterSha256: string | null;
    error: string | null;
    promptsPath: string | null;
    trainingCheckpoints: { jobId: string; step: number; epoch: number | null; path: string; statePath: string }[];
  }

  interface NativeLoraState {
    entries: NativeLora[];
    preparing: boolean;
    adaptersDir: string;
    converterSha256?: string;
  }

  interface LoraCard {
    name: string;
    legacy: Sa3LoraEntry | null;
    native: NativeLora | null;
  }

  let {
    open,
    serviceStatus,
    pendingRestart = false,
    onClose,
  }: {
    open: boolean;
    serviceStatus: "stopped" | "starting" | "running" | "unhealthy" | "failed";
    serviceEnvExists: boolean;
    pendingRestart?: boolean;
    onClose: () => void;
  } = $props();

  let loraState: Sa3LoraState | null = $state(null);
  let nativeState: NativeLoraState | null = $state(null);
  let nativeRuntime = $state(false);
  // The installed runtime's converter. Only a full load reports it; update
  // events leave it alone.
  let currentConverter: string | null = $state(null);
  let loading = $state(false);
  let busy = $state(false);
  // The LoRA an action is working on, or one of these list-wide actions.
  // Neither can be a LoRA name, which never contains "+".
  const ADDING = "+add";
  const PREPARING = "+prepare";
  let busyName = $state<string | null>(null);
  let message: string | null = $state(null);
  let error: string | null = $state(null);
  let buildOutput: string | null = $state(null);

  let legacySteps = $state<Record<string, number>>({});
  let nativeCheckpoints = $state<Record<string, string>>({});
  let strengths = $state<Record<string, number>>({});

  let formName = $state("");
  let sourcePath = $state("");
  let configPath = $state("");
  let promptsPath = $state("");

  function describeError(cause: unknown): string {
    return cause instanceof Error ? cause.message : String(cause);
  }

  function isGguf(path: string): boolean {
    return /\.gguf$/i.test(path.trim());
  }

  function suggestName(path: string): string {
    const parts = path.replace(/\\/g, "/").replace(/\/+$/, "").split("/");
    return (parts[parts.length - 1] || "")
      .replace(/\.(gguf|ckpt|safetensors)$/i, "")
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 64);
  }

  function applyLegacy(state: Sa3LoraState) {
    loraState = state;
  }

  function applyNative(state: NativeLoraState) {
    nativeState = state;
    strengths = Object.fromEntries(state.entries.map((entry) => [entry.name, entry.strength]));
  }

  // `keepNotes`: an action's outcome stays on screen across its refresh.
  async function loadState(keepNotes = false) {
    loading = true;
    if (!keepNotes) error = null;
    try {
      const [legacy, native, services] = await Promise.all([
        invoke<Sa3LoraState>("get_sa3_lora_state"),
        // A legacy entry the native registry cannot take fails only this
        // half: the list still shows, with that reason.
        invoke<NativeLoraState>("get_sa3_native_lora_state").catch((cause) => {
          error ??= describeError(cause);
          return null;
        }),
        invoke<{ id: string; runtime: string }[]>("get_services"),
      ]);
      applyLegacy(legacy);
      if (native) {
        applyNative(native);
        currentConverter = native.converterSha256 ?? null;
      }
      nativeRuntime = services.some((service) => service.id === "sa3" && service.runtime === "native");
    } catch (cause) {
      error = describeError(cause);
    } finally {
      loading = false;
    }
  }

  // Run one action: clear the notes, mark the LoRA busy, and refresh both
  // registries afterwards, since an action on one can change the other.
  async function run(name: string | null, action: () => Promise<string | void>) {
    busy = true;
    busyName = name;
    error = null;
    message = null;
    buildOutput = null;
    try {
      const done = await action();
      if (done) message = done;
    } catch (cause) {
      error = describeError(cause);
    } finally {
      busy = false;
      busyName = null;
      await loadState(true);
    }
  }

  async function pickSource() {
    const key = nativeRuntime ? "sa3-native-lora-source" : "sa3-lora-checkpoint";
    const selected = await openDialog({
      directory: false,
      multiple: false,
      defaultPath: rememberedDialogDirectory(key),
      filters: [{ name: "SA3 LoRA", extensions: nativeRuntime ? ["gguf", "safetensors", "ckpt"] : ["ckpt", "safetensors"] }],
    });
    if (typeof selected !== "string") return;
    rememberDialogSelection(key, selected, "file");
    sourcePath = selected;
    if (!formName.trim()) formName = suggestName(selected);
  }

  async function pickConfig() {
    const selected = await openDialog({
      directory: false,
      multiple: false,
      defaultPath: rememberedDialogDirectory("sa3-native-lora-config"),
      filters: [{ name: "LoRA configuration", extensions: ["json"] }],
    });
    if (typeof selected !== "string") return;
    rememberDialogSelection("sa3-native-lora-config", selected, "file");
    configPath = selected;
  }

  async function pickPrompts() {
    const selected = await openDialog({
      directory: true,
      multiple: false,
      defaultPath: rememberedDialogDirectory("sa3-lora-prompts"),
    });
    if (typeof selected !== "string") return;
    rememberDialogSelection("sa3-lora-prompts", selected, "directory");
    promptsPath = selected;
    if (!formName.trim()) formName = suggestName(selected);
  }

  function addLora() {
    const name = formName.trim();
    void run(ADDING, async () => {
      if (nativeRuntime) {
        await invoke<NativeLoraState>("import_sa3_native_lora", {
          name,
          checkpointPath: sourcePath,
          configPath: isGguf(sourcePath) ? null : configPath.trim() || null,
          promptsPath: promptsPath.trim() || null,
        });
      } else {
        await invoke<Sa3LoraState>("upsert_sa3_lora", {
          name,
          checkpointPath: sourcePath,
          promptsPath: promptsPath.trim() || null,
        });
      }
      formName = "";
      sourcePath = "";
      configPath = "";
      promptsPath = "";
      return `added ${name.toLowerCase()}. the source file is preserved.`;
    });
  }

  function prepare() {
    void run(PREPARING, async () => {
      const state = await invoke<NativeLoraState>("prepare_sa3_native_loras");
      const prepared = state.entries.filter((entry) => entry.nativePath && !entry.error).length;
      return `${prepared} of ${state.entries.length} LoRAs ready for C++. originals are preserved.`;
    });
  }

  function buildAllPrompts() {
    void run(null, async () => {
      const result = await invoke<Sa3PromptsBuildResult>("build_sa3_lora_prompts");
      buildOutput = result.output;
      return "prompt pools rebuilt";
    });
  }

  function rebuildPrompts(card: LoraCard) {
    void run(card.name, async () => {
      await invoke("update_sa3_native_lora", { name: card.name, change: { action: "build_prompts" } });
      return `rebuilt ${card.name}'s prompts`;
    });
  }

  function saveStrength(card: LoraCard) {
    void run(card.name, async () => {
      await invoke("update_sa3_native_lora", { name: card.name, change: { action: "strength", value: strengths[card.name] } });
      return `saved ${card.name}'s strength`;
    });
  }

  function legacyStep(card: LoraCard): number {
    const legacy = card.legacy!;
    return legacySteps[card.name]
      ?? legacy.selectedTrainingStep
      ?? legacy.trainingCheckpoints[legacy.trainingCheckpoints.length - 1]?.step
      ?? 0;
  }

  function useCheckpoint(card: LoraCard) {
    void run(card.name, async () => {
      if (usesLegacyCheckpoints(card)) {
        const step = legacyStep(card);
        await invoke("activate_sa3_lora_checkpoint", { name: card.name, step });
        return `${card.name} now uses training step ${step}${nativeRuntime ? ". prepare it again for C++." : ""}`;
      }
      await invoke("select_sa3_native_checkpoint", { name: card.name, checkpoint: nativeCheckpoints[card.name] });
      return `${card.name} now uses the chosen checkpoint. training originals are preserved.`;
    });
  }

  function remove(card: LoraCard) {
    void run(card.name, async () => {
      if (card.legacy) {
        await invoke("remove_sa3_lora", { name: card.name });
      } else {
        await invoke("update_sa3_native_lora", { name: card.name, change: { action: "remove" } });
      }
      return `removed ${card.name}. its files are preserved.`;
    });
  }

  function deleteFiles(card: LoraCard) {
    const confirmed = window.confirm(
      `Permanently delete '${card.name}' and all files from its Gary training run?\n\n` +
      `This deletes the managed LoRA, saved checkpoints, training cache/logs, and generated prompt JSON. ` +
      `Your original audio dataset and sidecars will not be changed.`
    );
    if (!confirmed) return;
    void run(card.name, async () => {
      await invoke("delete_sa3_trained_lora", { name: card.name });
      return `deleted ${card.name} and its Gary training files`;
    });
  }

  async function reveal(path: string) {
    try {
      await invoke("reveal_path", { path });
    } catch (cause) {
      error = describeError(cause);
    }
  }

  function usesLegacyCheckpoints(card: LoraCard): boolean {
    return (card.legacy?.trainingCheckpoints.length ?? 0) > 1;
  }

  function canRemove(card: LoraCard): boolean {
    return !!card.legacy || !!card.native?.nativeOnly || isGguf(card.native?.sourcePath ?? "");
  }

  // Converted by an earlier runtime's converter: preparing converts it again.
  function outdated(native: NativeLora | null): boolean {
    return !!native?.nativePath && !!native.converterSha256 && !!currentConverter
      && native.converterSha256 !== currentConverter;
  }

  function readiness(card: LoraCard): { label: string; tone: "ok" | "pending" | "bad"; title?: string } {
    if (card.native?.error) return { label: "failed to prepare", tone: "bad" };
    if (outdated(card.native)) return {
      label: "needs preparing",
      tone: "pending",
      title: "an earlier sa3.cpp converted this one. prepare it again to pick up the new runtime's fixes",
    };
    if (card.native?.nativePath) return { label: nativeRuntime ? "ready" : "ready for C++", tone: "ok" };
    if (card.legacy && !card.legacy.registered) return { label: "checkpoint missing", tone: "bad" };
    return { label: nativeRuntime ? "needs preparing" : "not prepared for C++", tone: "pending" };
  }

  let cards: LoraCard[] = $derived.by(() => {
    const byName = new Map<string, LoraCard>();
    for (const entry of loraState?.entries ?? []) byName.set(entry.name, { name: entry.name, legacy: entry, native: null });
    for (const entry of nativeState?.entries ?? []) {
      const card = byName.get(entry.name);
      if (card) card.native = entry;
      else byName.set(entry.name, { name: entry.name, legacy: null, native: entry });
    }
    return [...byName.values()].sort((a, b) => a.name.localeCompare(b.name));
  });

  let unprepared = $derived(cards.filter((card) => card.native && (!card.native.nativePath || card.native.error || outdated(card.native))).length);
  let legacyCkptPending = $derived(cards.some((card) => card.native && !card.native.nativePath && /\.ckpt$/i.test(card.native.sourcePath)));
  let sa3Active = $derived(["starting", "running", "unhealthy"].includes(serviceStatus));
  let blocked = $derived.by(() => loading || busy || pendingRestart || !!nativeState?.preparing || (nativeRuntime && sa3Active));

  $effect(() => {
    if (open) void loadState();
  });

  // Preparation reports each adapter as it goes.
  $effect(() => {
    if (!open) return;
    const stop = listen<NativeLoraState>("sa3-native-loras-updated", (event) => applyNative(event.payload));
    return () => void stop.then((unlisten) => unlisten());
  });
</script>

{#if open}
  <div class="overlay">
    <button type="button" class="backdrop" aria-label="close sa3 lora manager" onclick={onClose}></button>
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="sa3-lora-title" tabindex="-1">
      <div class="eyebrow">sa3 lora manager</div>
      <div class="title" id="sa3-lora-title">SA3 LoRAs</div>
      <div class="body">
        add a LoRA file, and optionally the dataset folder it was trained on: its txt sidecars become the plugin's
        prompt dice for that LoRA.
        {nativeRuntime
          ? "SA3 must be stopped while you change LoRAs. refresh the plugin's LoRA menu afterwards."
          : "SA3 loads LoRAs at startup: restart it to use a new or changed one."}
      </div>

      {#if pendingRestart}<div class="warning">restart gary4local to use the selected storage before changing LoRAs.</div>{/if}
      {#if nativeRuntime && sa3Active}<div class="warning">stop SA3 to add, prepare or change LoRAs.</div>{/if}

      <div class="section-label">add a LoRA</div>
      <div class="form-grid">
        <label class="field">
          <span>name</span>
          <input type="text" bind:value={formName} placeholder="my-lora" disabled={blocked} />
        </label>
        <label class="field">
          <span>LoRA file</span>
          <div class="path-row">
            <input type="text" bind:value={sourcePath} disabled={blocked}
              placeholder={nativeRuntime ? "GGUF, safetensors or CKPT" : "C:\\path\\to\\your.ckpt"} />
            <button type="button" onclick={pickSource} disabled={blocked}>pick file</button>
          </div>
        </label>
        <label class="field">
          <span>dataset folder for prompts (optional)</span>
          <div class="path-row">
            <input type="text" bind:value={promptsPath} disabled={blocked} placeholder="C:\\path\\to\\training dataset" />
            <button type="button" onclick={pickPrompts} disabled={blocked}>pick folder</button>
          </div>
        </label>
        {#if nativeRuntime && sourcePath.trim() && !isGguf(sourcePath)}
          <details>
            <summary>advanced: separate configuration file</summary>
            <div class="note">
              most adapters carry their configuration or have a matching JSON beside them, which gary finds itself.
              choose one only when your safetensors or CKPT export needs a separate model/LoRA configuration.
            </div>
            <div class="path-row">
              <input type="text" bind:value={configPath} disabled={blocked} placeholder="use the configuration supplied with your adapter" />
              <button type="button" onclick={pickConfig} disabled={blocked}>pick JSON</button>
            </div>
          </details>
        {/if}
      </div>
      <div class="actions">
        <button class="accent" onclick={addLora} disabled={blocked || !formName.trim() || !sourcePath.trim()}>
          {busyName === ADDING ? "adding..." : "add LoRA"}
        </button>
      </div>

      {#if message}<div class="success-note" role="status">{message}</div>{/if}
      {#if error}<div class="error-note" role="alert">{error}</div>{/if}
      {#if buildOutput}<pre class="output">{buildOutput}</pre>{/if}

      <div class="list-header">
        <div class="section-label">your LoRAs{cards.length ? ` (${cards.length})` : ""}</div>
        <div class="actions">
          {#if unprepared > 0}
            <button class="accent" onclick={prepare} disabled={blocked}
              title="convert LoRAs for the C++ runtime. originals are preserved">
              {nativeState?.preparing || busyName === PREPARING ? "preparing..." : `prepare ${unprepared} for C++`}
            </button>
          {/if}
          <button onclick={buildAllPrompts} disabled={blocked} title="rebuild every LoRA's prompt pool from its dataset folder">
            rebuild all prompts
          </button>
          <button onclick={() => void loadState()} disabled={loading || busy}>refresh</button>
          <button onclick={onClose}>close</button>
        </div>
      </div>
      {#if legacyCkptPending}
        <div class="note">
          .ckpt LoRAs are converted with SA3's Python environment: prepare them before cleaning up Python.
        </div>
      {/if}

      {#if loading && !loraState}
        <div class="empty">loading...</div>
      {:else if cards.length === 0}
        <div class="empty">no LoRAs yet.</div>
      {:else}
        <div class="entry-list">
          {#each cards as card (card.name)}
            {@const status = readiness(card)}
            {@const prompts = loraState?.pools[card.name] ?? card.legacy?.promptCount ?? 0}
            {@const promptSource = card.legacy?.promptsPath ?? card.native?.promptsPath ?? null}
            <div class="entry-card">
              <div class="entry-top">
                <div class="entry-heading">
                  <span class="entry-name">{card.name}</span>
                  <span class="chip {status.tone}" title={status.title}>{status.label}</span>
                  {#if card.legacy?.trainingJobId || card.native?.trainingCheckpoints.length}
                    <span class="chip">gary-trained</span>
                  {/if}
                </div>
                <div class="entry-actions">
                  {#if canRemove(card)}
                    <button class="danger" onclick={() => remove(card)} disabled={blocked}
                      title="take it off the list. its files are preserved">remove</button>
                  {/if}
                  {#if card.legacy?.trainingJobId}
                    <button class="danger delete-files" onclick={() => deleteFiles(card)} disabled={blocked}>
                      delete files
                    </button>
                  {/if}
                </div>
              </div>

              <div class="controls">
                {#if usesLegacyCheckpoints(card)}
                  <label class="inline-field">
                    <span>checkpoint</span>
                    <select value={legacyStep(card)} disabled={blocked}
                      onchange={(event) => (legacySteps = { ...legacySteps, [card.name]: Number(event.currentTarget.value) })}>
                      {#each card.legacy!.trainingCheckpoints as checkpoint}
                        <option value={checkpoint.step} title={checkpoint.path}>
                          step {checkpoint.step} / epoch {checkpoint.epoch}{checkpoint.step === card.legacy!.selectedTrainingStep ? " (in use)" : ""}
                        </option>
                      {/each}
                    </select>
                  </label>
                  <button onclick={() => useCheckpoint(card)}
                    disabled={blocked || legacyStep(card) === card.legacy!.selectedTrainingStep}>use</button>
                {:else if card.native?.trainingCheckpoints.length}
                  <label class="inline-field">
                    <span>checkpoint</span>
                    <select bind:value={nativeCheckpoints[card.name]} disabled={blocked}>
                      <option value="">choose a checkpoint</option>
                      {#each card.native.trainingCheckpoints as checkpoint (checkpoint.path)}
                        <option value={checkpoint.path}>
                          step {checkpoint.step}{checkpoint.epoch === null ? "" : ` / epoch ${checkpoint.epoch}`}{checkpoint.path === card.native.sourcePath ? " (in use)" : ""}
                        </option>
                      {/each}
                    </select>
                  </label>
                  <button onclick={() => useCheckpoint(card)} disabled={blocked || !nativeCheckpoints[card.name]}>use</button>
                {/if}

                {#if card.native}
                  <label class="inline-field strength">
                    <span>strength</span>
                    <input type="number" min="-4" max="4" step="0.05" bind:value={strengths[card.name]} disabled={blocked} />
                  </label>
                  <button onclick={() => saveStrength(card)}
                    disabled={blocked || !Number.isFinite(strengths[card.name]) || strengths[card.name] === card.native.strength}>save</button>
                {:else}
                  <span class="entry-meta">strength {card.legacy?.strength}</span>
                {/if}

                <span class="entry-meta prompts">
                  {promptSource ? `${prompts} prompts` : "no prompt dataset: the plugin's dice use defaults"}
                </span>
                {#if promptSource && card.native}
                  <button onclick={() => rebuildPrompts(card)} disabled={blocked}
                    title="rebuild this LoRA's prompt pool from its dataset folder">rebuild prompts</button>
                {/if}
              </div>

              {#if card.native?.error}<div class="error-note">{card.native.error}</div>{/if}

              <details class="paths">
                <summary>files</summary>
                {#if card.legacy || card.native}
                  {@const source = card.legacy?.path ?? card.native!.sourcePath}
                  <button type="button" class="path-link" onclick={() => void reveal(source)}>LoRA: {source}</button>
                {/if}
                {#if card.native?.nativePath}
                  <button type="button" class="path-link" onclick={() => void reveal(card.native!.nativePath!)}>C++ copy: {card.native.nativePath}</button>
                {/if}
                {#if promptSource}
                  <button type="button" class="path-link" onclick={() => void reveal(promptSource)}>prompt dataset: {promptSource}</button>
                {:else if card.legacy?.resolvedPromptsPath}
                  <div class="entry-path">prompt dataset: the sidecars beside the LoRA</div>
                {/if}
                {#if card.legacy?.promptFileExists}
                  <button type="button" class="path-link" onclick={() => void reveal(card.legacy!.promptFilePath)}>prompt pool: {card.legacy.promptFilePath}</button>
                {/if}
                {#if card.legacy?.trainingJobId}<div class="entry-path">training run: {card.legacy.trainingJobId}</div>{/if}
              </details>
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

  .field span,
  .inline-field span {
    font-size: 11px;
    color: var(--text-secondary);
  }

  .path-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 8px;
  }

  details summary {
    cursor: pointer;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 12px;
  }

  .accent {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }

  .accent:hover:not(:disabled) {
    background: var(--accent);
    border-color: var(--accent-hover);
  }

  .list-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
  }

  .list-header .actions {
    margin-top: 0;
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
    gap: 10px;
    margin-top: 12px;
  }

  .entry-card {
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.02);
    padding: 10px 12px;
  }

  .entry-top {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
  }

  .entry-heading {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    min-width: 0;
  }

  .entry-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .chip {
    font-size: 10px;
    padding: 2px 6px;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    white-space: nowrap;
  }

  .chip.ok {
    border-color: #3d6b48;
    color: #9bd8aa;
  }

  .chip.pending {
    border-color: #7a5a2e;
    color: #ffcb8f;
  }

  .chip.bad {
    border-color: #b24a4a;
    color: #ff8f8f;
  }

  .entry-actions {
    display: flex;
    gap: 8px;
    flex-shrink: 0;
  }

  .controls {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
  }

  .inline-field {
    display: grid;
    gap: 4px;
  }

  .inline-field select {
    min-width: 200px;
  }

  .strength input {
    width: 80px;
  }

  .prompts {
    margin: 0 0 6px 8px;
  }

  .paths {
    margin-top: 8px;
  }

  .path-link {
    display: block;
    width: 100%;
    margin-top: 4px;
    border: none;
    background: transparent;
    padding: 0;
    text-align: left;
    font: inherit;
    font-size: 11px;
    color: var(--text-secondary);
    cursor: pointer;
    word-break: break-word;
  }

  .path-link:hover,
  .path-link:focus-visible {
    text-decoration: underline;
  }

  .danger {
    border-color: #b24a4a;
    color: #ffb1b1;
  }

  .danger:hover:not(:disabled) {
    background: #7a2626;
    color: white;
  }

  .delete-files {
    background: #6f2020;
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
