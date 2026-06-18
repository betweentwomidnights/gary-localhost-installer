<script lang="ts">
  import { tick } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { open as openDialog } from "@tauri-apps/plugin-dialog";
  import CareyAceDatasetSidecarModal from "./CareyAceDatasetSidecarModal.svelte";

  interface CareyAceTrainingState {
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
    currentFile: number | null;
    totalFiles: number | null;
    sampleCount: number | null;
    captionedCount: number | null;
    captionLmModel: string | null;
    modelFamily: string | null;
    adapterType: string | null;
    captionsPath: string | null;
    datasetJsonPath: string | null;
    trainingPlanPath: string | null;
    resultPath: string | null;
    registeredLoraName: string | null;
    logTail: string;
  }

  let {
    open,
    serviceStatus,
    serviceEnvExists,
    onClose,
    onShowModels,
  }: {
    open: boolean;
    serviceStatus: "stopped" | "starting" | "running" | "unhealthy" | "failed";
    serviceEnvExists: boolean;
    onClose: () => void;
    onShowModels: () => void;
  } = $props();

  let trainingState: CareyAceTrainingState = $state({
    jobId: null,
    name: null,
    status: "idle",
    phase: "idle",
    message: "No ACE-Step LoRA training job has been started.",
    error: null,
    pid: null,
    childPid: null,
    runDir: null,
    logPath: null,
    cancelPath: null,
    finalCheckpointPath: null,
    currentStep: null,
    maxSteps: null,
    currentFile: null,
    totalFiles: null,
    sampleCount: null,
    captionedCount: null,
    captionLmModel: null,
    modelFamily: null,
    adapterType: null,
    captionsPath: null,
    datasetJsonPath: null,
    trainingPlanPath: null,
    resultPath: null,
    registeredLoraName: null,
    logTail: "",
  });

  let loading = $state(false);
  let starting = $state(false);
  let cancelling = $state(false);
  let error = $state<string | null>(null);
  let sidecarModalOpen = $state(false);
  let modalElement: HTMLDivElement | null = $state(null);
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
  let model = $state<"base" | "xl-base">("base");
  let adapterType = $state<"lora" | "dora">("dora");
  let trigger = $state("");
  let instrumental = $state(true);
  let autoCaption = $state(true);
  let captionLmModel = $state("acestep-5Hz-lm-1.7B");
  let overwriteCaptions = $state(false);
  let genreRatio = $state(20);
  let bpmAnalysis = $state(true);
  let keyAnalysis = $state(true);
  let includeAutoTimesignature = $state(false);
  let analysisDuration = $state(0);
  let rank = $state(64);
  let epochs = $state(150);
  let saveEvery = $state(25);
  let batchSize = $state(1);
  let gradientAccumulation = $state(1);
  let learningRateText = $state("3e-4");
  const cfgRatio = 0.15;
  let maxDuration = $state(240);

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

  function formatLearningRate(value: number): string {
    if (!Number.isFinite(value) || value <= 0) return "invalid";
    return value.toFixed(12).replace(/0+$/, "").replace(/\.$/, "");
  }

  async function pickDatasetFolder() {
    const selected = await openDialog({ directory: true, multiple: false });
    if (typeof selected !== "string") return;
    datasetPath = selected;
    if (!formName.trim()) {
      formName = suggestName(selected);
    }
    if (!trigger.trim()) {
      trigger = suggestName(selected);
    }
  }

  function handleLogMouseDown() {
    isSelectingLog = true;
  }

  function handleLogMouseUp() {
    setTimeout(() => {
      isSelectingLog = false;
    }, 100);
  }

  function nextFrame(): Promise<void> {
    return new Promise((resolve) => requestAnimationFrame(() => resolve()));
  }

  async function waitForLayout() {
    await tick();
    await nextFrame();
    await nextFrame();
  }

  function scrollLogPaneToBottom() {
    if (!logOutput) return;
    isLogAutoScrolling = true;
    logOutput.scrollTop = logOutput.scrollHeight;
    requestAnimationFrame(() => {
      isLogAutoScrolling = false;
    });
  }

  function scrollModalToBottom() {
    if (modalElement) {
      modalElement.scrollTop = modalElement.scrollHeight;
      return;
    }
    logSection?.scrollIntoView({ behavior: "auto", block: "end" });
  }

  async function scrollLogToBottom() {
    autoScrollLog = true;
    await waitForLayout();
    scrollLogPaneToBottom();
  }

  async function revealLogOutput() {
    shouldRevealLog = false;
    await waitForLayout();
    scrollLogPaneToBottom();
    scrollModalToBottom();
    await nextFrame();
    scrollLogPaneToBottom();
    scrollModalToBottom();
  }

  function handleLogScroll() {
    if (!logOutput || isLogAutoScrolling) return;
    const { scrollTop, scrollHeight, clientHeight } = logOutput;
    if (scrollHeight - scrollTop - clientHeight >= 50) {
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
      trainingState = await invoke<CareyAceTrainingState>("get_carey_ace_lora_training_state");
    } catch (e) {
      error = describeError(e);
    } finally {
      loading = false;
    }
  }

  async function startJob(prepareOnly: boolean) {
    starting = true;
    error = null;
    autoScrollLog = true;
    try {
      trainingState = await invoke<CareyAceTrainingState>("start_carey_ace_lora_training", {
        name: formName,
        datasetPath,
        model,
        adapterType,
        trigger,
        instrumental,
        autoCaption: prepareOnly ? autoCaption : false,
        captionLmModel,
        overwriteCaptions,
        genreRatio,
        prepareOnly,
        includeAutoTimesignature,
        bpmAnalysis,
        keyAnalysis,
        analysisDuration,
        rank,
        epochs,
        saveEvery,
        batchSize,
        gradientAccumulation,
        learningRate,
        cfgRatio,
        maxDuration,
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
      trainingState = await invoke<CareyAceTrainingState>("cancel_carey_ace_lora_training");
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
  let commonValid = $derived(
    open &&
      serviceEnvExists &&
      !starting &&
      !cancelling &&
      !isTraining &&
      !!formName.trim() &&
      !!datasetPath.trim() &&
      rank > 0 &&
      epochs > 0 &&
      saveEvery > 0 &&
      batchSize > 0 &&
      gradientAccumulation > 0 &&
      maxDuration > 0 &&
      genreRatio >= 0 &&
      genreRatio <= 100 &&
      Number.isFinite(learningRate) &&
      learningRate > 0 &&
      Number.isFinite(cfgRatio) &&
      cfgRatio >= 0 &&
      cfgRatio < 1
  );
  let canPrepare = $derived(commonValid);
  let canTrain = $derived(commonValid);

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
      if (jobId && isTraining && !starting) {
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
        scrollModalToBottom();
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
    <button type="button" class="backdrop" aria-label="close ace-step lora trainer" onclick={onClose}></button>
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="ace-train-title" tabindex="-1" bind:this={modalElement}>
      <div class="eyebrow">carey ace-step trainer</div>
      <div class="title" id="ace-train-title">train a local ACE-Step LoRA</div>
      <div class="body">
        prepare ACE sidecars with understand_music, edit the metadata, then train a LoRA or DoRA adapter against base or xl-base. Successful runs register themselves in Carey's LoRA registry.
        <div class="upstream-credit">
          trainer references koda-dernet's
          <button type="button" onclick={() => void invoke("open_url", { url: "https://github.com/koda-dernet/Side-Step" })}>Side-Step</button>
          project, adapted into Gary's local workflow.
        </div>
      </div>

      {#if !serviceEnvExists}
        <div class="warning">build Carey first so the ACE-Step training environment exists.</div>
      {:else if serviceStatus === "running"}
        <div class="warning">Starting the trainer will stop Carey inference first so the captioner/trainer can own GPU memory.</div>
      {/if}

      <div class="section-label">dataset</div>
      <div class="form-grid">
        <label class="field">
          <span>LoRA name</span>
          <input type="text" bind:value={formName} placeholder="my-ace-style" />
        </label>
        <label class="field">
          <span>trigger tag</span>
          <input type="text" bind:value={trigger} placeholder="patch" />
        </label>

        <label class="field wide">
          <span>audio folder</span>
          <div class="path-row">
            <input type="text" bind:value={datasetPath} placeholder="C:\\path\\to\\ace dataset" />
            <button type="button" onclick={pickDatasetFolder}>pick folder</button>
            <button type="button" onclick={() => sidecarModalOpen = true} disabled={!datasetPath.trim()}>
              edit prompts / sidecars
            </button>
          </div>
        </label>
      </div>

      <div class="section-label">captioner</div>
      <div class="settings-grid">
        <label class="field">
          <span>LM model</span>
          <select bind:value={captionLmModel}>
            <option value="acestep-5Hz-lm-0.6B">0.6B low VRAM</option>
            <option value="acestep-5Hz-lm-1.7B">1.7B recommended</option>
            <option value="acestep-5Hz-lm-4B">4B large GPU</option>
          </select>
        </label>
        <label class="toggle-field">
          <input type="checkbox" bind:checked={autoCaption} />
          <span>
            <strong>auto-caption missing sidecars</strong>
            <small>starts a temporary LM-backed ACE captioner for full-track understand_music, then releases GPU memory.</small>
          </span>
        </label>
        <label class="toggle-field">
          <input type="checkbox" bind:checked={overwriteCaptions} />
          <span>
            <strong>overwrite existing sidecars</strong>
            <small>use this only when you want to replace human edits.</small>
          </span>
        </label>
        <label class="field">
          <span>genre prompt ratio</span>
          <input type="number" min="0" max="100" step="1" bind:value={genreRatio} />
          <small>Each epoch, {genreRatio}% of tracks use genre instead of caption.</small>
        </label>
        <label class="toggle-field">
          <input type="checkbox" bind:checked={bpmAnalysis} />
          <span>
            <strong>BPM sanity check</strong>
            <small>local tempo estimate can override obvious LM mistakes.</small>
          </span>
        </label>
        <label class="toggle-field">
          <input type="checkbox" bind:checked={keyAnalysis} />
          <span>
            <strong>key sanity check</strong>
            <small>conservative local chroma estimate; ambiguous keys stay editable.</small>
          </span>
        </label>
        <label class="toggle-field">
          <input type="checkbox" bind:checked={includeAutoTimesignature} />
          <span>
            <strong>include LM signature</strong>
            <small>off by default because signatures are easy to hallucinate and optional.</small>
          </span>
        </label>
        <label class="toggle-field">
          <input type="checkbox" bind:checked={instrumental} />
          <span>
            <strong>instrumental default</strong>
            <small>empty lyrics become [Instrumental]; pasted lyrics still win.</small>
          </span>
        </label>
      </div>
      <div class="captioner-prepare">
        <div>
          <strong>caption and prepare dataset</strong>
          {#if autoCaption}
            <small>runs understand_music for missing sidecars, applies BPM/key checks, then writes dataset metadata for review.</small>
          {:else}
            <small>skips LM captioning and prepares dataset metadata from your existing editable sidecars.</small>
          {/if}
        </div>
        <button type="button" class="accent" onclick={() => void startJob(true)} disabled={!canPrepare}>
          {starting ? "launching..." : "caption / prepare"}
        </button>
      </div>

      <div class="section-label">training</div>
      <div class="settings-grid">
        <label class="field">
          <span>base model</span>
          <select bind:value={model}>
            <option value="base">ace-step-v15-base</option>
            <option value="xl-base">ace-step-v15-xl-base</option>
          </select>
        </label>
        <label class="field">
          <span>adapter</span>
          <select bind:value={adapterType}>
            <option value="dora">DoRA</option>
            <option value="lora">LoRA</option>
          </select>
        </label>
        <label class="field">
          <span>rank</span>
          <input type="number" min="1" step="1" bind:value={rank} />
        </label>
        <label class="field">
          <span>epochs</span>
          <input type="number" min="1" step="1" bind:value={epochs} />
        </label>
        <label class="field">
          <span>save every</span>
          <input type="number" min="1" step="1" bind:value={saveEvery} />
        </label>
        <label class="field">
          <span>batch size</span>
          <input type="number" min="1" step="1" bind:value={batchSize} />
        </label>
        <label class="field">
          <span>grad accum</span>
          <input type="number" min="1" step="1" bind:value={gradientAccumulation} />
        </label>
        <label class="field">
          <span>learning rate</span>
          <input type="text" inputmode="decimal" spellcheck="false" bind:value={learningRateText} />
          <small class:invalid={learningRateDecimal === "invalid"}>decimal: {learningRateDecimal}</small>
        </label>
        <label class="field">
          <span>max track seconds</span>
          <input type="number" min="1" step="1" bind:value={maxDuration} />
        </label>
      </div>

      <div class="note">
        Use caption/prepare first, review the sidecars, then train; the trainer owns its captioning process separately from Carey inference.
        <button type="button" class="inline-link" onclick={onShowModels}>open Carey models</button>
      </div>

      <div class="actions">
        <button type="button" class="accent-secondary" onclick={() => void startJob(false)} disabled={!canTrain}>
          {starting ? "launching..." : `train ${adapterType === "dora" ? "DoRA" : "LoRA"}`}
        </button>
        {#if isTraining}
          <button type="button" class="danger" onclick={cancelTraining} disabled={cancelling}>
            {cancelling ? "cancelling..." : "cancel job"}
          </button>
        {/if}
        <button type="button" onclick={() => void loadTrainingState()} disabled={loading}>refresh</button>
        <button type="button" onclick={onClose}>close</button>
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
                {#if trainingState.currentFile !== null && trainingState.totalFiles}
                  / file {trainingState.currentFile} of {trainingState.totalFiles}
                {:else if trainingState.currentStep !== null && trainingState.maxSteps}
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
          {#if trainingState.captionedCount !== null}
            <div class="success-note">captioned {trainingState.captionedCount} track{trainingState.captionedCount === 1 ? "" : "s"}</div>
          {/if}
          {#if trainingState.finalCheckpointPath}
            <button type="button" class="success-note path-link" onclick={() => void revealPath(trainingState.finalCheckpointPath!)} title="Show checkpoint in folder">
              registered checkpoint: {trainingState.finalCheckpointPath}
            </button>
          {/if}
          {#if trainingState.datasetJsonPath}
            <button type="button" class="job-path path-link" onclick={() => void revealPath(trainingState.datasetJsonPath!)} title="Show dataset in folder">
              dataset: {trainingState.datasetJsonPath}
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
        <div class="log-wrap" role="presentation" onmousedown={handleLogMouseDown} onmouseup={handleLogMouseUp}>
          <pre class="output" bind:this={logOutput} tabindex="-1" onscroll={handleLogScroll}>{trainingState?.logTail || "No output yet."}</pre>
        </div>
      </div>
    </div>
  </div>
  <CareyAceDatasetSidecarModal
    open={sidecarModalOpen}
    {datasetPath}
    sharedTrigger={trigger}
    defaultInstrumental={instrumental}
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
    z-index: 74;
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
    width: min(840px, 100%);
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

  .upstream-credit {
    margin-top: 8px;
    color: var(--text-secondary);
  }

  .upstream-credit button,
  .inline-link {
    border: none;
    background: transparent;
    color: var(--text-primary);
    font: inherit;
    font-weight: 600;
    padding: 0;
    text-decoration: underline;
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

  .form-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
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

  .path-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto;
    gap: 8px;
  }

  .toggle-field {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    color: var(--text-primary);
    cursor: pointer;
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

  .captioner-prepare {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-top: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.03);
    padding: 12px;
  }

  .captioner-prepare div {
    display: grid;
    gap: 4px;
  }

  .captioner-prepare strong {
    color: var(--text-primary);
    font-size: 12px;
    font-weight: 600;
  }

  .captioner-prepare small {
    color: var(--text-muted);
    font: 10px var(--font-mono);
    line-height: 1.45;
  }

  .captioner-prepare .accent {
    flex: 0 0 auto;
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }

  .actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 16px;
  }

  .actions .accent-secondary {
    border-color: var(--accent);
    color: var(--text-primary);
  }

  .actions .danger {
    border-color: rgba(255, 120, 120, 0.72);
    background: rgba(155, 42, 42, 0.28);
    color: #ffb3b3;
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
    height: clamp(200px, 28vh, 260px);
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

  @media (max-width: 760px) {
    .overlay {
      padding: 12px;
    }

    .modal {
      padding: 16px;
      max-height: 94vh;
    }

    .path-row,
    .form-grid,
    .settings-grid {
      grid-template-columns: 1fr;
    }

    .captioner-prepare {
      align-items: stretch;
      flex-direction: column;
    }
  }
</style>
