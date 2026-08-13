<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";

  interface Sa3DatasetSidecarEntry {
    audioPath: string;
    relativePath: string;
    sidecarPath: string;
    content: string;
    exists: boolean;
    jsonSidecarExists: boolean;
  }

  interface DraftEntry extends Sa3DatasetSidecarEntry {
    originalContent: string;
  }

  interface Sa3DatasetSidecarSaveResult {
    saved: number;
    removed: number;
    entries: Sa3DatasetSidecarEntry[];
  }

  interface Sa3MetadataSuggestion {
    bpm: number | null;
    keyscale: string;
    suggestion: string;
    bpmConfidence: number | null;
    keyConfidence: number | null;
  }

  interface Sa3AutolabelState {
    status: string; // "" (idle) | starting | running | completed | cancelled | failed
    phase: string;
    message: string;
    total: number;
    done: number;
    currentPath: string;
    style: string;
    error: string | null;
    pid: number | null;
  }

  interface Sa3AutolabelAvailability {
    available: boolean;
    careyBuilt: boolean;
    captionerDownloaded: boolean;
    analysisModelsDownloaded: boolean;
  }

  type PromptStyle = "bare" | "labeled";

  let {
    open,
    datasetPath,
    onClose,
  }: {
    open: boolean;
    datasetPath: string;
    onClose: () => void;
  } = $props();

  // BPM and key are stripped from the dice pool and re-added by the plugin from its
  // own dropdowns, so either style trains fine. Bare is the default because it matches
  // what gary4juce appends at inference.
  const barePrompt = "technical death metal, 145 bpm, C minor";
  const labeledPrompt =
    "TrackType: Music, VocalType: Instrumental, Genre: technical death metal, Mood: absurd, BPM: 145, Key: C minor";

  let entries: DraftEntry[] = $state([]);
  let selectedIndex = $state(0);
  let templateText = $state(barePrompt);
  let promptStyle = $state<PromptStyle>("bare");
  let showStyleInfo = $state(false);
  let loading = $state(false);
  let saving = $state(false);
  let suggesting = $state(false);
  let autolabelAvailable = $state(false);
  let autolabelCareyBuilt = $state(false);
  let autolabelCaptionerDownloaded = $state(false);
  let autolabelAnalysisModelsDownloaded = $state(false);
  let autolabelState = $state<Sa3AutolabelState | null>(null);
  let autolabelPollTimer: ReturnType<typeof setInterval> | null = null;
  let lastAutolabelDone = 0; // completed count at the previous poll, to pull focus once per finish
  let trackListEl = $state<HTMLElement>();
  let error = $state<string | null>(null);
  // A single transient note, tagged with where it should render so it sits inline
  // next to its trigger (fill/save button, suggest row) instead of adding a row.
  let note = $state<string | null>(null);
  let noteContext = $state<"fill" | "save" | "suggest" | null>(null);
  let loadedPath = $state("");
  let wasOpen = false;

  function describeError(value: unknown): string {
    return value instanceof Error ? value.message : String(value);
  }

  function clearNote() {
    note = null;
    noteContext = null;
  }

  async function openReference(reference: "underfit" | "prompting") {
    error = null;
    try {
      await invoke("open_sa3_training_reference", { reference });
    } catch (e) {
      error = describeError(e);
    }
  }

  // Mirror of prompt_from_caption in services/sa3/build_lora_prompts.py: peel any
  // trailing bpm/key tag (labeled "BPM: 145" / "Key: C minor" or bare "145 bpm" /
  // "C minor") until the tail is stable, so both are removed in any order.
  const diceTrailingTag =
    /[,;]?\s*(?:bpm\s*[:=]?\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*bpm|(?:key|scale)\s*[:=]\s*[A-G][#b♯♭]?\s+(?:maj(?:or)?|min(?:or)?)|(?<![A-Za-z])[A-G][#b♯♭]?\s+(?:major|minor))\s*$/i;

  function dicePromptFromCaption(text: string): string {
    let prompt = text.trim();
    for (;;) {
      const next = prompt.replace(diceTrailingTag, "").replace(/^[,;\s]+|[,;\s]+$/g, "");
      if (next === prompt) return prompt;
      prompt = next;
    }
  }

  // Switch the fill style. Swap the starter text too, but only when it's still one of
  // the known examples, so a user's hand-edited prompt is never clobbered.
  function setPromptStyle(style: PromptStyle) {
    const current = templateText.trim();
    if (current === barePrompt || current === labeledPrompt) {
      templateText = style === "bare" ? barePrompt : labeledPrompt;
    }
    promptStyle = style;
  }

  function formatMetadataTag(bpm: number | null, keyscale: string): string {
    const parts: string[] = [];
    if (bpm != null) parts.push(promptStyle === "labeled" ? `BPM: ${bpm}` : `${bpm} bpm`);
    if (keyscale) parts.push(promptStyle === "labeled" ? `Key: ${keyscale}` : keyscale);
    return parts.join(", ");
  }

  // Idempotent: strip any trailing bpm/key first (same peel as the dice preview), then
  // append the freshly formatted tag, so re-pressing or switching style never stacks.
  function spliceMetadata(content: string, tag: string): string {
    const base = dicePromptFromCaption(content);
    if (!tag) return base;
    if (!base) return tag;
    return `${base}, ${tag}`;
  }

  async function suggestBpmKey() {
    const entry = entries[selectedIndex];
    if (!entry) return;
    suggesting = true;
    error = null;
    clearNote();
    try {
      const result = await invoke<Sa3MetadataSuggestion>("suggest_sa3_track_metadata", {
        datasetPath,
        audioPath: entry.audioPath,
      });
      const tag = formatMetadataTag(result.bpm, result.keyscale);
      if (!tag) {
        note = "no bpm or key detected";
        noteContext = "suggest";
        return;
      }
      // No success toast: the updated prompt textarea is its own confirmation, and a
      // note here would steal height and push the suggest button past the scroll fold.
      entries[selectedIndex] = { ...entry, content: spliceMetadata(entry.content, tag) };
    } catch (e) {
      error = describeError(e);
    } finally {
      suggesting = false;
    }
  }

  function toDrafts(items: Sa3DatasetSidecarEntry[]): DraftEntry[] {
    return items.map((item) => ({ ...item, originalContent: item.content }));
  }

  // silent: refresh entries without the loading state or clearing error/notes — used
  // to re-sync txt/none labels after an auto-label run without flashing the editor.
  async function loadSidecars(silent = false) {
    if (!datasetPath.trim()) return;
    if (!silent) {
      loading = true;
      error = null;
      clearNote();
    }
    try {
      const result = await invoke<Sa3DatasetSidecarEntry[]>("get_sa3_dataset_sidecars", {
        datasetPath,
      });
      entries = toDrafts(result);
      selectedIndex = Math.min(selectedIndex, Math.max(0, entries.length - 1));
      loadedPath = datasetPath;
    } catch (e) {
      error = describeError(e);
    } finally {
      if (!silent) loading = false;
    }
  }

  function stopAutolabelPolling() {
    if (autolabelPollTimer !== null) {
      clearInterval(autolabelPollTimer);
      autolabelPollTimer = null;
    }
  }

  function startAutolabelPolling() {
    stopAutolabelPolling();
    autolabelPollTimer = setInterval(() => void refreshAutolabelState(), 1500);
  }

  async function refreshAutolabelState() {
    try {
      const state = await invoke<Sa3AutolabelState>("get_sa3_autolabel_state");
      autolabelState = state;
      const running = state.status === "starting" || state.status === "running";
      const terminal = ["completed", "cancelled", "failed"].includes(state.status);
      if (running || terminal) {
        // Re-sync from disk each tick: the job writes each sidecar as it finishes, so
        // completed rows flip none -> txt live rather than all at once at the end.
        await loadSidecars(true);
      }
      // Pull focus once, right after a track finishes — to the just-completed track
      // (which now has its prompt), and only on the completion transition so that
      // navigating away between finishes isn't hijacked.
      if (state.done > lastAutolabelDone) {
        lastAutolabelDone = state.done;
        selectedIndex = Math.min(state.done - 1, entries.length - 1);
      }
      if (terminal || !state.status || state.status === "idle") {
        stopAutolabelPolling();
        if (state.status === "failed" && state.error) error = state.error;
      }
    } catch (e) {
      stopAutolabelPolling();
      error = describeError(e);
    }
  }

  // Called when the modal opens: probe availability and resume polling if a job is
  // already running (e.g. the modal was closed and reopened mid-run).
  async function initAutolabel() {
    try {
      const availability = await invoke<Sa3AutolabelAvailability>(
        "get_sa3_autolabel_availability"
      );
      autolabelAvailable = availability.available;
      autolabelCareyBuilt = availability.careyBuilt;
      autolabelCaptionerDownloaded = availability.captionerDownloaded;
      autolabelAnalysisModelsDownloaded = availability.analysisModelsDownloaded;
    } catch {
      autolabelAvailable = false;
      autolabelCareyBuilt = false;
      autolabelCaptionerDownloaded = false;
      autolabelAnalysisModelsDownloaded = false;
    }
    try {
      autolabelState = await invoke<Sa3AutolabelState>("get_sa3_autolabel_state");
      if (autolabelState.status === "starting" || autolabelState.status === "running") {
        lastAutolabelDone = autolabelState.done; // resume without yanking focus to an old finish
        startAutolabelPolling();
      }
    } catch {
      /* no job yet */
    }
  }

  async function startAutolabel() {
    error = null;
    clearNote();
    if (dirtyCount) {
      error = "Save or discard your unsaved sidecar changes before auto-labeling.";
      return;
    }
    if (!autolabelAvailable) {
      error = !autolabelCareyBuilt
        ? "Build Carey before auto-labeling."
        : !autolabelCaptionerDownloaded
          ? "Download the ACE-Step 5Hz LM 1.7B captioner from Carey → Models before auto-labeling."
          : "Download ACE-Step Base, VAE, and Qwen3 Embedding from Carey → Models before auto-labeling.";
      return;
    }
    lastAutolabelDone = 0;
    try {
      autolabelState = await invoke<Sa3AutolabelState>("start_sa3_autolabel", {
        datasetPath,
        style: promptStyle,
      });
      startAutolabelPolling();
    } catch (e) {
      error = describeError(e);
    }
  }

  async function cancelAutolabel() {
    try {
      autolabelState = await invoke<Sa3AutolabelState>("cancel_sa3_autolabel");
    } catch (e) {
      error = describeError(e);
    }
  }

  function renderedTemplate(): string {
    // Sidecars hold captions only. The shared trigger word is a separate layer the
    // trainer prepends to every caption (see build_dataset_config), so it is
    // deliberately not part of prompt editing or the dice pool.
    return templateText.trim();
  }

  function fillMissing() {
    const template = renderedTemplate();
    if (!template) {
      error = "Enter a template first.";
      return;
    }
    error = null;
    let filled = 0;
    entries = entries.map((entry) => {
      if (entry.content.trim() || entry.jsonSidecarExists) return entry;
      filled += 1;
      return { ...entry, content: template };
    });
    // No success toast: the none -> txt flips in the track list are the cue. Only the
    // "nothing to fill" case needs a word, and it renders inline beside the button.
    if (filled) {
      clearNote();
    } else {
      note = "no empty sidecars to fill";
      noteContext = "fill";
    }
  }

  // clear empties the draft and saves immediately, so the sidecar is removed from disk
  // rather than lingering until "save sidecars" (which had surprised us before).
  async function clearCurrent() {
    const entry = entries[selectedIndex];
    if (!entry) return;
    entries[selectedIndex] = { ...entry, content: "" };
    await saveSidecars();
  }

  async function clearAll() {
    if (!entries.length) return;
    entries = entries.map((entry) => ({ ...entry, content: "" }));
    await saveSidecars();
  }

  async function saveSidecars() {
    const changed = entries.filter((entry) => entry.content !== entry.originalContent);
    if (!changed.length) {
      note = "no changes to save";
      noteContext = "save";
      return;
    }

    saving = true;
    error = null;
    clearNote();
    try {
      const result = await invoke<Sa3DatasetSidecarSaveResult>("save_sa3_dataset_sidecars", {
        datasetPath,
        sidecars: changed.map((entry) => ({
          audioPath: entry.audioPath,
          content: entry.content,
        })),
      });
      entries = toDrafts(result.entries);
      selectedIndex = Math.min(selectedIndex, Math.max(0, entries.length - 1));
      const details = [];
      if (result.saved) details.push(`${result.saved} saved`);
      if (result.removed) details.push(`${result.removed} removed`);
      note = details.length ? `Sidecars updated: ${details.join(", ")}.` : "Sidecars are already current.";
      noteContext = "save";
    } catch (e) {
      error = describeError(e);
    } finally {
      saving = false;
    }
  }

  function selectTrack(index: number) {
    selectedIndex = index;
    clearNote();
  }

  let selectedEntry = $derived(entries[selectedIndex] ?? null);
  let dirtyCount = $derived(
    entries.filter((entry) => entry.content !== entry.originalContent).length
  );
  let captionedCount = $derived(entries.filter((entry) => entry.content.trim()).length);
  let selectedDicePrompt = $derived(
    selectedEntry ? dicePromptFromCaption(selectedEntry.content) : ""
  );
  let autolabelRunning = $derived(
    !!autolabelState &&
      (autolabelState.status === "starting" || autolabelState.status === "running")
  );
  // Files are processed in the same relative-path order the modal lists them, so the
  // in-progress row is simply the one at the completed count.
  // The reused Carey polling re-labels the phase as "captioning" mid-track, so treat
  // both it and our own "analyzing" as the active analysis state.
  let analyzingIndex = $derived(
    autolabelRunning &&
      (autolabelState?.phase === "analyzing" || autolabelState?.phase === "captioning")
      ? (autolabelState?.done ?? -1)
      : -1
  );
  let autolabelTitle = $derived(
    !autolabelCareyBuilt
      ? "Requires Carey — build Carey first."
      : !autolabelCaptionerDownloaded
        ? "Download the ACE-Step 5Hz LM 1.7B captioner from Carey → Models first."
        : !autolabelAnalysisModelsDownloaded
          ? "Download ACE-Step Base, VAE, and Qwen3 Embedding from Carey → Models first."
          : dirtyCount
            ? "Save or discard unsaved sidecar changes before auto-labeling."
            : "Auto-label every track's genre, BPM and key (overwrites existing sidecars). Runs the Carey caption model."
  );

  $effect(() => {
    const justOpened = open && !wasOpen;
    wasOpen = open;
    if (!open) {
      stopAutolabelPolling();
      return;
    }
    if (!datasetPath.trim()) return;
    if (justOpened || datasetPath !== loadedPath) {
      void loadSidecars();
      void initAutolabel();
    }
  });

  $effect(() => () => stopAutolabelPolling());

  // Keep the selected row visible in the track list so the auto-label focus pull can
  // scroll to a finished track that's below the fold. No-ops when it's already shown.
  $effect(() => {
    const idx = selectedIndex;
    if (!open || idx < 0) return;
    trackListEl
      ?.querySelector<HTMLElement>(".track-row.active")
      ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
</script>

{#if open}
  <div class="sidecar-overlay">
    <button type="button" class="sidecar-backdrop" aria-label="close dataset prompt editor" onclick={onClose}></button>
    <div class="sidecar-modal" role="dialog" aria-modal="true" aria-labelledby="sidecar-title">
      <div class="header">
        <div>
          <div class="eyebrow">optional dataset prompts</div>
          <div class="title" id="sidecar-title">edit SA3 text sidecars</div>
        </div>
        <div class="header-actions">
          <button type="button" onclick={() => loadSidecars()} disabled={loading || saving || autolabelRunning}>refresh</button>
          <button type="button" onclick={onClose}>close</button>
        </div>
      </div>

      <div class="body">
        Give each audio file an optional same-name `.txt` file, such as `song.wav` and `song.txt`. Everything in the text file is used as that track's prompt.
      </div>

      <div class="template-band">
        <div class="style-row">
          <span>prompt style</span>
          <div class="style-toggle">
            <button type="button" class:active={promptStyle === "bare"} onclick={() => setPromptStyle("bare")} disabled={autolabelRunning}>barebones</button>
            <button type="button" class:active={promptStyle === "labeled"} onclick={() => setPromptStyle("labeled")} disabled={autolabelRunning}>official SA3</button>
          </div>
          <button type="button" class="info-toggle" aria-label="about prompt styles" aria-expanded={showStyleInfo} onclick={() => (showStyleInfo = !showStyleInfo)}>ⓘ</button>
        </div>
        {#if showStyleInfo}
          <small class="style-info">
            Personally I just use barebones — <code>genre, 145 bpm, C minor</code>. The official SA3 repo recommends a labeled style — <code>BPM: 145, Key: C minor</code> — see the <button type="button" class="reference-link" onclick={() => void openReference("prompting")}>official SA3 prompting guide</button>. Either trains fine: BPM and key are stripped from the dice pool and the plugin re-adds them from its own dropdowns.
          </small>
        {/if}
        <label class="template-field">
          <span>editable starter prompt — {promptStyle === "bare" ? "barebones" : "official SA3"} style</span>
          <textarea rows="2" bind:value={templateText}></textarea>
        </label>
        <div class="template-actions">
          <div class="action-buttons">
            <button type="button" onclick={fillMissing} disabled={loading || !entries.length || autolabelRunning}>fill missing</button>
            {#if autolabelRunning}
              <button type="button" class="autolabel-cancel" onclick={cancelAutolabel}>
                cancel auto-label{autolabelState?.total ? ` (${autolabelState.done}/${autolabelState.total})` : ""}
              </button>
            {:else}
              <button
                type="button"
                onclick={startAutolabel}
                disabled={!autolabelAvailable || loading || saving || suggesting || !!dirtyCount || !entries.length}
                title={autolabelTitle}
              >auto-label all</button>
            {/if}
            <button
              type="button"
              onclick={clearAll}
              disabled={loading || saving || autolabelRunning || !entries.length}
              title="Remove every track's sidecar (saves immediately)"
            >clear all</button>
          </div>
          <span class:inline-note={noteContext === "fill"}>
            {#if autolabelRunning}
              {autolabelState?.message || "Auto-labeling…"}
            {:else}
              {noteContext === "fill" && note ? note : `${captionedCount} of ${entries.length} tracks have prompt text`}
            {/if}
          </span>
        </div>
      </div>

      {#if error}
        <div class="error-note">{error}</div>
      {/if}

      {#if loading}
        <div class="empty">Scanning audio files...</div>
      {:else if !entries.length}
        <div class="empty">No supported audio files found in this folder.</div>
      {:else}
        <div class="editor">
          <div class="track-list" bind:this={trackListEl}>
            {#each entries as entry, index}
              <button
                type="button"
                class:active={index === selectedIndex}
                class="track-row"
                onclick={() => selectTrack(index)}
              >
                <span class="track-name">{entry.relativePath}</span>
                <span class:filled={!!entry.content.trim()} class="track-state">
                  {#if index === analyzingIndex}
                    <span class="spinner" aria-label="analyzing"></span>
                  {:else}
                    {entry.content.trim() ? "txt" : "none"}
                  {/if}
                </span>
              </button>
            {/each}
          </div>

          {#if selectedEntry}
            <div class="track-editor">
              <div class="track-editor-scroll">
                <div class="track-heading">
                  <div>
                    <div class="track-title">{selectedEntry.relativePath}</div>
                    <div class="sidecar-path">{selectedEntry.sidecarPath}</div>
                  </div>
                  <span>{selectedIndex + 1} / {entries.length}</span>
                </div>

                {#if selectedEntry.jsonSidecarExists}
                  <div class="warning">A JSON sidecar exists for this track and takes precedence over `.txt` during pre-encoding.</div>
                {/if}

                <label class="prompt-field">
                  <span>literal text-sidecar prompt</span>
                  <textarea rows="3" bind:value={selectedEntry.content} placeholder="Leave blank to train without a per-track prompt." disabled={autolabelRunning}></textarea>
                </label>
                <div class="metadata-assist">
                  <button type="button" class="assist-button" onclick={suggestBpmKey} disabled={suggesting || saving || autolabelRunning}>
                    {suggesting ? "analyzing…" : "suggest bpm / key"}
                  </button>
                  <small class:inline-note={noteContext === "suggest"}>{noteContext === "suggest" && note ? note : `Estimates tempo and key from the audio and fills them in ${promptStyle === "bare" ? "barebones" : "official SA3"} style. First use may pause briefly to install an analysis dependency.`}</small>
                </div>
                <div class="dice-preview">
                  <span>dice button result</span>
                  <div>{selectedDicePrompt || "not added to the LoRA prompt pool"}</div>
                  {#if selectedEntry.content.trim() !== selectedDicePrompt}
                    <small>Trailing BPM and key tags are omitted — the plugin adds those from its own dropdowns.</small>
                  {/if}
                </div>
              </div>

              <div class="track-actions">
                <button type="button" onclick={clearCurrent} disabled={saving || autolabelRunning}>clear</button>
                <button
                  type="button"
                  onclick={() => selectedIndex = Math.max(0, selectedIndex - 1)}
                  disabled={selectedIndex === 0}
                >previous</button>
                <button
                  type="button"
                  onclick={() => selectedIndex = Math.min(entries.length - 1, selectedIndex + 1)}
                  disabled={selectedIndex === entries.length - 1}
                >next</button>
              </div>
            </div>
          {/if}
        </div>
      {/if}

      <div class="footer">
        <span class:inline-note={noteContext === "save" && !dirtyCount}>
          {dirtyCount
            ? `${dirtyCount} unsaved change${dirtyCount === 1 ? "" : "s"}`
            : noteContext === "save" && note
              ? note
              : "all changes saved"}
        </span>
        <button type="button" class="accent" onclick={saveSidecars} disabled={saving || autolabelRunning || !dirtyCount}>
          {saving ? "saving..." : "save sidecars"}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .sidecar-overlay {
    position: fixed;
    inset: 0;
    z-index: 76;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }

  .sidecar-backdrop {
    position: absolute;
    inset: 0;
    border: none;
    background: rgba(0, 0, 0, 0.82);
    padding: 0;
  }

  .sidecar-modal {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    width: min(920px, 100%);
    max-height: min(92vh, 900px);
    border: 1px solid var(--border);
    background: var(--bg-secondary);
    box-shadow: 0 22px 64px rgba(0, 0, 0, 0.58);
    overflow: hidden;
  }

  /* Only the editor flexes/scrolls; the fixed rows keep their height so the
     footer stays pinned and the panes get a bounded, scrollable region. */
  .header,
  .body,
  .template-band,
  .error-note,
  .footer {
    flex-shrink: 0;
  }

  .header,
  .footer,
  .track-heading,
  .template-actions,
  .track-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
  }

  .header {
    padding: 16px 18px;
    border-bottom: 1px solid var(--border);
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  .eyebrow,
  .template-field span,
  .prompt-field span {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-secondary);
  }

  .title {
    margin-top: 5px;
    font-size: 18px;
    font-weight: 600;
  }

  .body {
    padding: 12px 18px 0;
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 1.5;
  }

  .reference-link {
    border: none;
    background: transparent;
    color: var(--text-primary);
    font-weight: 600;
    padding: 0;
    text-decoration: underline;
  }

  .template-band {
    display: grid;
    gap: 9px;
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
  }

  .template-field,
  .prompt-field {
    display: grid;
    gap: 6px;
  }

  .style-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .style-row > span {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-secondary);
  }

  .style-toggle {
    display: inline-flex;
    border: 1px solid var(--border);
  }

  .style-toggle button {
    border: none;
    background: transparent;
    color: var(--text-secondary);
    padding: 4px 12px;
    font-size: 11px;
  }

  .style-toggle button.active {
    background: var(--bg-panel);
    color: var(--text-primary);
    box-shadow: inset 0 -2px 0 var(--accent);
  }

  .info-toggle {
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-secondary);
    width: 22px;
    height: 22px;
    padding: 0;
    line-height: 1;
    border-radius: 50%;
  }

  .style-info {
    color: var(--text-secondary);
    font: 11px/1.5 var(--font-mono);
  }

  .style-info code {
    color: var(--text-primary);
  }

  .metadata-assist {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 12px;
    flex-wrap: wrap;
  }

  .metadata-assist small {
    color: var(--text-secondary);
    font-size: 10px;
    line-height: 1.4;
    flex: 1;
    min-width: 180px;
  }

  .assist-button {
    background: var(--bg-panel);
    border: 1px solid var(--accent);
    color: var(--text-primary);
    padding: 6px 14px;
    font-size: 12px;
    white-space: nowrap;
  }

  .assist-button:disabled {
    opacity: 0.6;
  }

  textarea {
    width: 100%;
    resize: vertical;
    border: 1px solid var(--border);
    background: var(--bg-primary);
    color: var(--text-primary);
    font: 12px/1.5 var(--font-mono);
    padding: 9px;
    user-select: text;
    -webkit-user-select: text;
  }

  .template-actions span,
  .footer span,
  .track-heading > span {
    color: var(--text-secondary);
    font-size: 11px;
  }

  .editor {
    display: grid;
    grid-template-columns: minmax(220px, 0.38fr) minmax(0, 1fr);
    grid-template-rows: minmax(0, 1fr);
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
  }

  .track-list {
    overflow: auto;
    min-height: 0;
    border-right: 1px solid var(--border);
    background: var(--bg-primary);
  }

  .track-row {
    width: 100%;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: 8px;
    border: none;
    border-bottom: 1px solid var(--border);
    background: transparent;
    padding: 9px 10px;
    text-align: left;
    border-radius: 0;
  }

  .track-row.active {
    background: var(--bg-panel);
    box-shadow: inset 3px 0 0 var(--accent);
  }

  .track-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11px;
  }

  .track-state {
    color: var(--text-muted);
    font: 10px var(--font-mono);
  }

  .track-state.filled {
    color: var(--green);
  }

  .action-buttons {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .autolabel-cancel {
    border-color: var(--accent);
    color: var(--text-primary);
  }

  .spinner {
    display: inline-block;
    width: 11px;
    height: 11px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: sa3-spin 0.7s linear infinite;
    vertical-align: middle;
  }

  @keyframes sa3-spin {
    to {
      transform: rotate(360deg);
    }
  }

  .track-editor {
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    overflow: hidden;
  }

  .track-editor-scroll {
    flex: 1 1 auto;
    min-height: 0;
    overflow: auto;
    padding: 16px;
  }

  .track-title {
    font-size: 14px;
    font-weight: 600;
    word-break: break-word;
  }

  .sidecar-path {
    margin-top: 4px;
    color: var(--text-muted);
    font: 10px/1.4 var(--font-mono);
    word-break: break-word;
    user-select: text;
    -webkit-user-select: text;
  }

  .prompt-field {
    margin-top: 14px;
  }

  .dice-preview {
    display: grid;
    gap: 5px;
    margin-top: 10px;
    padding: 9px;
    border-left: 2px solid var(--accent);
    background: var(--bg-primary);
    font: 11px/1.45 var(--font-mono);
    word-break: break-word;
    user-select: text;
    -webkit-user-select: text;
  }

  .dice-preview span,
  .dice-preview small {
    color: var(--text-secondary);
    font: 10px/1.4 var(--font-mono);
  }

  .track-actions {
    flex-shrink: 0;
    justify-content: flex-end;
    flex-wrap: wrap;
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    background: var(--bg-panel);
  }

  .warning,
  .error-note,
  .empty {
    margin: 10px 18px 0;
    font-size: 11px;
    line-height: 1.45;
  }

  .warning {
    margin: 12px 0 0;
    color: #ffcb8f;
  }

  .error-note {
    color: #ff8f8f;
  }

  .inline-note {
    color: var(--green);
  }

  .empty {
    padding: 22px 0;
    color: var(--text-secondary);
  }

  .footer {
    margin-top: auto;
    padding: 12px 18px;
    border-top: 1px solid var(--border);
    background: var(--bg-panel);
  }

</style>
