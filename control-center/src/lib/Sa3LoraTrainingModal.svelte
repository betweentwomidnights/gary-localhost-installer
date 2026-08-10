<script lang="ts">
  import { tick } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { open as openDialog } from "@tauri-apps/plugin-dialog";
  import Sa3DatasetSidecarModal from "./Sa3DatasetSidecarModal.svelte";

  interface Sa3LoraTrainingState {
    jobId: string | null;
    name: string | null;
    status: string;
    phase: string;
    message: string;
    error: string | null;
    pid: number | null;
    childPid: number | null;
    runDir: string | null;
    logPath: string | null;
    cancelPath: string | null;
    finalCheckpointPath: string | null;
    currentStep: number | null;
    maxSteps: number | null;
    logTail: string;
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

  let trainingState: Sa3LoraTrainingState = $state({
    jobId: null,
    name: null,
    status: "idle",
    phase: "idle",
    message: "No SA3 LoRA training job has been started.",
    error: null,
    pid: null,
    childPid: null,
    runDir: null,
    logPath: null,
    cancelPath: null,
    finalCheckpointPath: null,
    currentStep: null,
    maxSteps: null,
    logTail: "",
  });
  let starting = $state(false);
  let cancelling = $state(false);
  let error = $state<string | null>(null);
  let sidecarModalOpen = $state(false);
  let logSection: HTMLDivElement | null = $state(null);
  let logOutput: HTMLPreElement | null = $state(null);
  let autoScrollLog = $state(true);
  let lastJobId: string | null = $state(null);
  let lastLogLength = $state(0);
  let isLogAutoScrolling = false;
  let isSelectingLog = false;
  let shouldRevealLog = false;

  let formName = $state("");
  let datasetPath = $state("");
  let fixedPrompt = $state("");
  let maxSteps = $state(2000);
  let rank = $state(16);
  let batchSize = $state(1);
  let checkpointEvery = $state(500);
  // Full-track training (spark / stable-audio-3 default): train each clip whole,
  // starting at its downbeat, so the model keeps its native duration/bpm =>
  // seamless-loop behavior. 380s clamps to the model's native 4096 latent
  // tokens. Turn off to train shorter random crops on low-VRAM cards.
  let fullTrack = $state(true);
  // 285.35s = the model's real native length (3072 latent tokens). The model
  // config's "380s" is wrong per the Spark notes; 380 would pad every clip out to
  // 4096 tokens (up to ~52% padding on short clips). 285.35 matches the Spark's
  // proven duration and its ~12% padding — more correct AND faster.
  const FULL_TRACK_CROP_SECONDS = 285.35;
  let latentCropSeconds = $state(47);
  let effectiveCropSeconds = $derived(fullTrack ? FULL_TRACK_CROP_SECONDS : latentCropSeconds);
  let learningRateText = $state("1e-4");
  let loudnessFixEnabled = $state(false);
  let targetLatentRms = $state(0.9);
  // The efficient MLX-compatible scope keeps the seven attention/feed-forward
  // projections in each of the 24 transformer blocks: 7 * 24 = 168 adapters.
  let layerScope = $state("transformer-core");

  function describeError(value: unknown): string {
    return value instanceof Error ? value.message : String(value);
  }

  async function revealPath(path: string) {
    error = null;
    try {
      await invoke("reveal_path", { path });
    } catch (e) {
      error = describeError(e);
    }
  }

  function formatLearningRate(value: number): string {
    if (!Number.isFinite(value) || value <= 0) return "invalid";
    return value.toFixed(12).replace(/0+$/, "").replace(/\.$/, "");
  }

  async function openUnderfit() {
    error = null;
    try {
      await invoke("open_sa3_training_reference", { reference: "underfit" });
    } catch (e) {
      error = describeError(e);
    }
  }

  function basename(path: string): string {
    const normalized = path.replace(/\\/g, "/").replace(/\/+$/, "");
    const parts = normalized.split("/");
    return parts[parts.length - 1] || "";
  }

  function suggestName(path: string): string {
    return basename(path)
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 64);
  }

  function handleLogMouseDown() {
    isSelectingLog = true;
  }

  function handleLogMouseUp() {
    setTimeout(() => {
      isSelectingLog = false;
    }, 100);
  }

  async function scrollLogToBottom() {
    autoScrollLog = true;
    await tick();
    if (logOutput) {
      isLogAutoScrolling = true;
      logOutput.scrollTop = logOutput.scrollHeight;
      requestAnimationFrame(() => {
        isLogAutoScrolling = false;
      });
    }
  }

  async function revealLogOutput() {
    await tick();
    logSection?.scrollIntoView({ behavior: "smooth", block: "start" });
    await scrollLogToBottom();
  }

  function handleLogScroll() {
    if (!logOutput || isLogAutoScrolling) return;
    const { scrollTop, scrollHeight, clientHeight } = logOutput;
    const nearBottom = scrollHeight - scrollTop - clientHeight < 50;
    if (!nearBottom) {
      autoScrollLog = false;
    }
  }

  function selectLogOutput() {
    if (!logOutput) return;
    autoScrollLog = false;
    isSelectingLog = true;
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(logOutput);
    selection?.removeAllRanges();
    selection?.addRange(range);
    logOutput.focus();
  }

  async function loadTrainingState() {
    try {
      trainingState = await invoke<Sa3LoraTrainingState>("get_sa3_lora_training_state");
    } catch (e) {
      error = describeError(e);
    }
  }

  async function pickDatasetFolder() {
    const selected = await openDialog({ directory: true, multiple: false });
    if (typeof selected !== "string") return;
    datasetPath = selected;
    if (!formName.trim()) {
      formName = suggestName(selected);
    }
  }

  async function startTraining() {
    starting = true;
    error = null;
    autoScrollLog = true;
    try {
      trainingState = await invoke<Sa3LoraTrainingState>("start_sa3_lora_training", {
        name: formName,
        datasetPath,
        fixedPrompt,
        maxSteps,
        rank,
        batchSize,
        checkpointEvery,
        latentCropSeconds: effectiveCropSeconds,
        learningRate,
        loudnessFixEnabled,
        targetLatentRms,
        layerScope,
      });
      shouldRevealLog = true;
      await revealLogOutput();
    } catch (e) {
      error = describeError(e);
    } finally {
      starting = false;
    }
  }

  async function cancelTraining() {
    cancelling = true;
    error = null;
    try {
      trainingState = await invoke<Sa3LoraTrainingState>("cancel_sa3_lora_training");
    } catch (e) {
      error = describeError(e);
    } finally {
      cancelling = false;
    }
  }

  let isTraining = $derived(
    trainingState?.status === "starting" || trainingState?.status === "running"
  );
  let learningRate = $derived(Number(learningRateText.trim()));
  let learningRateDecimal = $derived(formatLearningRate(learningRate));
  let canStart = $derived(
    open &&
      serviceEnvExists &&
      serviceStatus !== "running" &&
      !starting &&
      !cancelling &&
      !isTraining &&
      !!formName.trim() &&
      !!datasetPath.trim() &&
      maxSteps > 0 &&
      rank > 0 &&
      batchSize > 0 &&
      checkpointEvery > 0 &&
      (fullTrack || latentCropSeconds > 0) &&
      Number.isFinite(learningRate) &&
      learningRate > 0 &&
      (!loudnessFixEnabled ||
        (Number.isFinite(targetLatentRms) &&
          targetLatentRms >= 0.5 &&
          targetLatentRms <= 1.3))
  );

  $effect(() => {
    if (!open) return;
    void loadTrainingState();
    const timer = window.setInterval(() => {
      void loadTrainingState();
    }, 3000);
    return () => window.clearInterval(timer);
  });

  $effect(() => {
    const jobId = trainingState?.jobId ?? null;
    if (jobId !== lastJobId) {
      lastJobId = jobId;
      lastLogLength = 0;
      autoScrollLog = true;
      if (jobId && isTraining) {
        shouldRevealLog = true;
      }
    }
  });

  $effect(() => {
    const currentLength = trainingState?.logTail?.length ?? 0;
    if (currentLength < lastLogLength) {
      autoScrollLog = true;
    }
    lastLogLength = currentLength;
  });

  $effect(() => {
    const logTail = trainingState?.logTail ?? "";
    const reveal = shouldRevealLog;
    if (!open || !logOutput || (!logTail && !reveal) || !autoScrollLog) return;
    if (isSelectingLog) return;

    const selection = window.getSelection();
    if (
      selection &&
      selection.rangeCount > 0 &&
      !selection.isCollapsed &&
      logOutput.contains(selection.anchorNode)
    ) {
      return;
    }

    isLogAutoScrolling = true;
    tick().then(() => {
      if (logOutput) {
        logOutput.scrollTop = logOutput.scrollHeight;
      }
      if (reveal) {
        logSection?.scrollIntoView({ behavior: "smooth", block: "start" });
        shouldRevealLog = false;
      }
      requestAnimationFrame(() => {
        isLogAutoScrolling = false;
      });
    });
  });
</script>

{#if open}
  <div class="overlay">
    <button type="button" class="backdrop" aria-label="close sa3 lora trainer" onclick={onClose}></button>
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="sa3-train-title" tabindex="-1">
      <div class="eyebrow">sa3 lora trainer</div>
      <div class="title" id="sa3-train-title">train a local SA3 LoRA</div>
      <div class="body">
        select a folder of audio files. optional same-name `.txt` prompts are picked up during encoding; the finished checkpoint is added to the SA3 LoRA registry.
        <div class="upstream-credit">
          this LoRA trainer is a stripped-down version of dada-bots'
          <button type="button" onclick={() => void openUnderfit()}>underfit</button>
          repo.
        </div>
      </div>

      {#if !serviceEnvExists}
        <div class="warning">build SA3 first so the training environment exists.</div>
      {:else if serviceStatus === "running"}
        <div class="warning">stop SA3 before training. generation keeps the model in VRAM.</div>
      {/if}

      <div class="section-label">dataset</div>
      <div class="form-grid">
        <label class="field">
          <span>LoRA name</span>
          <input type="text" bind:value={formName} placeholder="my-style" />
        </label>

        <label class="field wide">
          <span>audio folder</span>
          <div class="path-row">
            <input type="text" bind:value={datasetPath} placeholder="C:\\path\\to\\audio dataset" />
            <button type="button" onclick={pickDatasetFolder}>pick folder</button>
            <button type="button" onclick={() => sidecarModalOpen = true} disabled={!datasetPath.trim()}>
              edit prompts
            </button>
          </div>
        </label>

        <label class="field wide">
          <span>shared trigger word</span>
          <input type="text" bind:value={fixedPrompt} placeholder="optional — e.g. my-trigger" />
          <small>prepended to every caption during training ("trigger, caption"), so the word comes to mean this style. Leave blank to train on captions alone.</small>
        </label>
      </div>

      <div class="section-label">training</div>
      <div class="settings-grid">
        <label class="field">
          <span>steps</span>
          <input type="number" min="1" step="100" bind:value={maxSteps} />
        </label>
        <label class="field">
          <span>rank</span>
          <input type="number" min="1" step="1" bind:value={rank} />
        </label>
        <label class="field">
          <span>batch size</span>
          <input type="number" min="1" step="1" bind:value={batchSize} />
        </label>
        <label class="field">
          <span>checkpoint every</span>
          <input type="number" min="1" step="100" bind:value={checkpointEvery} />
        </label>
        <label class="toggle-field wide">
          <input type="checkbox" bind:checked={fullTrack} />
          <span>
            <strong>train on full tracks</strong>
            <small>trains each clip whole (up to ~380s) from its downbeat, matching the stable-audio-3 defaults — best for seamless loops and bpm-accurate lengths. Uses more VRAM; turn off to train shorter random crops on smaller cards.</small>
          </span>
        </label>
        {#if !fullTrack}
          <label class="field">
            <span>crop seconds</span>
            <input type="number" min="1" step="1" bind:value={latentCropSeconds} />
            <small>random crop length. Shorter = less VRAM, but loops start mid-beat.</small>
          </label>
        {/if}
        <label class="field">
          <span>learning rate</span>
          <input type="text" inputmode="decimal" spellcheck="false" bind:value={learningRateText} />
          <small class:invalid={learningRateDecimal === "invalid"}>
            decimal: {learningRateDecimal}
          </small>
        </label>
        <label class="field wide">
          <span>LoRA layer scope</span>
          <select bind:value={layerScope}>
            <option value="transformer-core">efficient transformer core (168 layers)</option>
            <option value="full">full reference scope (229 layers)</option>
            <option value="full-no-seconds">full DiT, no duration conditioner (228 layers)</option>
          </select>
          <small>
            168 trains every block's self-attention, cross-attention, and feed-forward projections while skipping local-conditioning and outer projection adapters. It matches the efficient MLX scope and produces a normal SA3 LoRA checkpoint. Use 229 only for exact reference-recipe parity.
          </small>
        </label>
        <label class="toggle-field wide">
          <input type="checkbox" bind:checked={loudnessFixEnabled} />
          <span>
            <strong>experimental loudness fix</strong>
            <small>normalizes each track's encoded latent RMS; pre-encoding will take longer.</small>
          </span>
        </label>
        {#if loudnessFixEnabled}
          <label class="field">
            <span>target latent RMS</span>
            <input type="number" min="0.5" max="1.3" step="0.01" bind:value={targetLatentRms} />
            <small>0.90 matches base-model loudness. Lower is quieter; higher is hotter.</small>
          </label>
        {/if}
      </div>

      <div class="note">
        defaults: DoRA, efficient 168-layer scope, bf16/fp16 frozen base weights, batch 1, full-track training, and no training demos. Full-track needs more VRAM — on 8-12 GB cards, turn off "train on full tracks" to use shorter random crops.
      </div>

      <div class="actions">
        <button class="accent" onclick={startTraining} disabled={!canStart}>
          {starting ? "launching..." : "train LoRA"}
        </button>
        {#if isTraining}
          <button class="danger" onclick={cancelTraining} disabled={cancelling}>
            {cancelling ? "cancelling..." : "cancel training"}
          </button>
        {/if}
        <button onclick={onClose}>close</button>
      </div>

      {#if error}
        <div class="error-note">{error}</div>
      {/if}

      <div class="section-label">current job</div>
      {#if trainingState}
        <div class="job-card">
          <div class="job-top">
            <div>
              <div class="job-name">{trainingState.name ?? trainingState.jobId ?? "idle"}</div>
              <div class="job-meta">
                {trainingState.status || "idle"} / {trainingState.phase || "idle"}
                {#if trainingState.currentStep !== null && trainingState.maxSteps}
                  / step {Math.min(trainingState.currentStep, trainingState.maxSteps)} of {trainingState.maxSteps}
                {/if}
              </div>
            </div>
            {#if isTraining}
              <div class="live">running</div>
            {/if}
          </div>
          {#if trainingState.error && trainingState.error.trim() === trainingState.message.trim()}
            <div class="error-note">{trainingState.error}</div>
          {:else}
            <div class="job-message">{trainingState.message}</div>
          {/if}
          {#if trainingState.error && trainingState.error.trim() !== trainingState.message.trim()}
            <div class="error-note">{trainingState.error}</div>
          {/if}
          {#if trainingState.finalCheckpointPath}
            <button type="button" class="success-note path-link" onclick={() => void revealPath(trainingState.finalCheckpointPath!)} title="Show checkpoint in folder">
              registered checkpoint: {trainingState.finalCheckpointPath}
            </button>
          {/if}
          {#if trainingState.runDir}
            <button type="button" class="job-path path-link" onclick={() => void revealPath(trainingState.runDir!)} title="Open run folder">
              run: {trainingState.runDir}
            </button>
          {/if}
          {#if trainingState.logPath}
            <button type="button" class="job-path path-link" onclick={() => void revealPath(trainingState.logPath!)} title="Show log in folder">
              log: {trainingState.logPath}
            </button>
          {/if}
        </div>
      {:else}
        <div class="empty">no training state yet.</div>
      {/if}

      <div class="log-section" bind:this={logSection}>
        <div class="log-header">
          <span>log output</span>
          <div class="log-actions">
            <button type="button" class="scroll-btn" onclick={selectLogOutput}>select all</button>
            {#if !autoScrollLog}
              <button type="button" class="scroll-btn" onclick={scrollLogToBottom}>scroll to bottom</button>
            {/if}
          </div>
        </div>
        <div
          class="log-wrap"
          role="presentation"
          onmousedown={handleLogMouseDown}
          onmouseup={handleLogMouseUp}
        >
          <pre
            class="output"
            bind:this={logOutput}
            tabindex="-1"
            onscroll={handleLogScroll}
          >{trainingState?.logTail || "No output yet."}</pre>
        </div>
      </div>
    </div>
  </div>
  <Sa3DatasetSidecarModal
    open={sidecarModalOpen}
    {datasetPath}
    onClose={() => sidecarModalOpen = false}
  />
{/if}

<style>
  .overlay {
    position: fixed;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    z-index: 72;
  }

  .backdrop {
    position: absolute;
    inset: 0;
    border: none;
    background: rgba(0, 0, 0, 0.72);
    padding: 0;
  }

  .upstream-credit {
    margin-top: 8px;
    color: var(--text-secondary);
  }

  .upstream-credit button {
    border: none;
    background: transparent;
    color: var(--text-primary);
    font: inherit;
    font-weight: 600;
    padding: 0;
    text-decoration: underline;
  }

  .modal {
    position: relative;
    z-index: 1;
    width: min(780px, 100%);
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
  .job-meta,
  .job-message,
  .job-path {
    margin-top: 8px;
    font-size: 11px;
    line-height: 1.45;
    word-break: break-word;
  }

  .note,
  .job-meta,
  .job-message,
  .job-path {
    color: var(--text-secondary);
  }

  .path-link {
    display: block;
    width: 100%;
    border: none;
    background: transparent;
    padding: 0;
    text-align: left;
    font: inherit;
    cursor: pointer;
  }

  .path-link:hover,
  .path-link:focus-visible {
    text-decoration: underline;
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

  .form-grid,
  .settings-grid {
    display: grid;
    gap: 12px;
    margin-top: 12px;
  }

  .settings-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .field {
    display: grid;
    gap: 6px;
  }

  .field span {
    font-size: 11px;
    color: var(--text-secondary);
  }

  .field small {
    color: var(--text-muted);
    font: 10px var(--font-mono);
  }

  .field small.invalid {
    color: var(--red);
  }

  .field.wide {
    grid-column: 1 / -1;
  }

  .toggle-field {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    color: var(--text-primary);
    cursor: pointer;
  }

  .toggle-field.wide {
    grid-column: 1 / -1;
  }

  .toggle-field input {
    width: 15px;
    height: 15px;
    margin: 1px 0 0;
    accent-color: var(--accent);
    flex: 0 0 auto;
  }

  .toggle-field span {
    display: grid;
    gap: 3px;
  }

  .toggle-field strong {
    font-size: 11px;
    font-weight: 600;
  }

  .toggle-field small {
    color: var(--text-muted);
    font: 10px var(--font-mono);
    line-height: 1.45;
  }

  .path-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto;
    gap: 8px;
  }

  .actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 16px;
  }

  .actions .accent {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }

  .actions .danger {
    border-color: rgba(255, 120, 120, 0.72);
    background: rgba(155, 42, 42, 0.28);
    color: #ffb3b3;
  }

  .log-section {
    margin-top: 12px;
  }

  .log-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    border: 1px solid var(--border);
    border-bottom: none;
    background: rgba(255, 255, 255, 0.03);
    padding: 6px 10px;
  }

  .log-header span {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-secondary);
    font-weight: 600;
  }

  .scroll-btn {
    font-size: 10px;
    padding: 2px 8px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 3px;
  }

  .log-actions {
    display: flex;
    gap: 6px;
  }

  .log-wrap {
    overflow: hidden;
  }

  .job-card {
    margin-top: 12px;
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.02);
    padding: 12px;
  }

  .job-top {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: flex-start;
  }

  .job-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .live {
    border: 1px solid var(--green);
    color: var(--green);
    padding: 2px 8px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }

  .output {
    margin: 0;
    padding: 10px;
    border: 1px solid var(--border);
    background: rgba(0, 0, 0, 0.24);
    color: var(--text-primary);
    font-size: 11px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 260px;
    overflow: auto;
    user-select: text;
    -webkit-user-select: text;
    cursor: text;
    outline: none;
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

    .path-row,
    .settings-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
