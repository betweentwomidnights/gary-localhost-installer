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
  let loading = $state(false);
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
  let latentCropSeconds = $state(47);
  let learningRateText = $state("1e-4");
  let loudnessFixEnabled = $state(false);
  let targetLatentRms = $state(0.9);

  function describeError(value: unknown): string {
    return value instanceof Error ? value.message : String(value);
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
    loading = true;
    try {
      trainingState = await invoke<Sa3LoraTrainingState>("get_sa3_lora_training_state");
    } catch (e) {
      error = describeError(e);
    } finally {
      loading = false;
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
        latentCropSeconds,
        learningRate,
        loudnessFixEnabled,
        targetLatentRms,
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
      latentCropSeconds > 0 &&
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
          <span>style prompt / trigger text</span>
          <input type="text" bind:value={fixedPrompt} placeholder="optional shared style phrase" />
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
        <label class="field">
          <span>crop seconds</span>
          <input type="number" min="1" step="1" bind:value={latentCropSeconds} />
        </label>
        <label class="field">
          <span>learning rate</span>
          <input type="text" inputmode="decimal" spellcheck="false" bind:value={learningRateText} />
          <small class:invalid={learningRateDecimal === "invalid"}>
            decimal: {learningRateDecimal}
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
        defaults favor 8-16 GB cards: DoRA, fp16 base weights, batch 1, random 47s crops, and no training demos.
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
        <button onclick={() => void loadTrainingState()} disabled={loading}>refresh</button>
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
          <div class="job-message">{trainingState.message}</div>
          {#if trainingState.error}
            <div class="error-note">{trainingState.error}</div>
          {/if}
          {#if trainingState.finalCheckpointPath}
            <div class="success-note">registered checkpoint: {trainingState.finalCheckpointPath}</div>
          {/if}
          {#if trainingState.runDir}
            <div class="job-path">run: {trainingState.runDir}</div>
          {/if}
          {#if trainingState.logPath}
            <div class="job-path">log: {trainingState.logPath}</div>
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
    sharedPrompt={fixedPrompt}
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
