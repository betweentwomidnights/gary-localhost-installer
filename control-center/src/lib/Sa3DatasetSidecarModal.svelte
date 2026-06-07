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

  let {
    open,
    datasetPath,
    sharedPrompt,
    onClose,
  }: {
    open: boolean;
    datasetPath: string;
    sharedPrompt: string;
    onClose: () => void;
  } = $props();

  const starterPrompt =
    "TrackType: Music, VocalType: Instrumental, Genre: technical death metal, Mood: absurd, BPM: 145";

  let entries: DraftEntry[] = $state([]);
  let selectedIndex = $state(0);
  let templateText = $state(starterPrompt);
  let includeSharedPrompt = $state(false);
  let loading = $state(false);
  let saving = $state(false);
  let error = $state<string | null>(null);
  let message = $state<string | null>(null);
  let loadedPath = $state("");
  let wasOpen = false;

  function describeError(value: unknown): string {
    return value instanceof Error ? value.message : String(value);
  }

  async function openReference(reference: "underfit" | "prompting") {
    error = null;
    try {
      await invoke("open_sa3_training_reference", { reference });
    } catch (e) {
      error = describeError(e);
    }
  }

  function dicePromptFromCaption(text: string): string {
    return text
      .replace(
        /(?:[,;]\s*)?(?:BPM\s*:\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*BPM)\s*$/i,
        ""
      )
      .trim()
      .replace(/^[,;\s]+|[,;\s]+$/g, "");
  }

  function toDrafts(items: Sa3DatasetSidecarEntry[]): DraftEntry[] {
    return items.map((item) => ({ ...item, originalContent: item.content }));
  }

  async function loadSidecars() {
    if (!datasetPath.trim()) return;
    loading = true;
    error = null;
    message = null;
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
      loading = false;
    }
  }

  function renderedTemplate(): string {
    const pieces = [];
    if (includeSharedPrompt && sharedPrompt.trim()) {
      pieces.push(sharedPrompt.trim());
    }
    if (templateText.trim()) {
      pieces.push(templateText.trim());
    }
    return pieces.join(", ");
  }

  function fillMissing() {
    const template = renderedTemplate();
    if (!template) {
      error = "Enter a template or enable the shared phrase first.";
      return;
    }
    error = null;
    let filled = 0;
    entries = entries.map((entry) => {
      if (entry.content.trim() || entry.jsonSidecarExists) return entry;
      filled += 1;
      return { ...entry, content: template };
    });
    message = filled
      ? `Filled ${filled} missing sidecar draft${filled === 1 ? "" : "s"}.`
      : "No empty text sidecars found. Tracks with JSON metadata were skipped.";
  }

  function clearCurrent() {
    const entry = entries[selectedIndex];
    if (!entry) return;
    entries[selectedIndex] = { ...entry, content: "" };
    message = null;
  }

  function restoreCurrent() {
    const entry = entries[selectedIndex];
    if (!entry) return;
    entries[selectedIndex] = { ...entry, content: entry.originalContent };
    message = null;
  }

  async function saveSidecars() {
    const changed = entries.filter((entry) => entry.content !== entry.originalContent);
    if (!changed.length) {
      message = "No sidecar changes to save.";
      return;
    }

    saving = true;
    error = null;
    message = null;
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
      message = details.length ? `Sidecars updated: ${details.join(", ")}.` : "Sidecars are already current.";
    } catch (e) {
      error = describeError(e);
    } finally {
      saving = false;
    }
  }

  function selectTrack(index: number) {
    selectedIndex = index;
    message = null;
  }

  let selectedEntry = $derived(entries[selectedIndex] ?? null);
  let dirtyCount = $derived(
    entries.filter((entry) => entry.content !== entry.originalContent).length
  );
  let captionedCount = $derived(entries.filter((entry) => entry.content.trim()).length);
  let selectedDicePrompt = $derived(
    selectedEntry ? dicePromptFromCaption(selectedEntry.content) : ""
  );

  $effect(() => {
    const justOpened = open && !wasOpen;
    wasOpen = open;
    if (!open || !datasetPath.trim()) return;
    if (justOpened || datasetPath !== loadedPath) {
      void loadSidecars();
    }
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
          <button type="button" onclick={loadSidecars} disabled={loading || saving}>refresh</button>
          <button type="button" onclick={onClose}>close</button>
        </div>
      </div>

      <div class="body">
        Give each audio file an optional same-name `.txt` file, such as `song.wav` and `song.txt`. Everything in the text file is used as that track's prompt.
        <div class="references">
          <button type="button" class="reference-link" onclick={() => void openReference("underfit")}>
            open Underfit metadata guide
          </button>
          <button type="button" class="reference-link" onclick={() => void openReference("prompting")}>
            open official SA3 prompting guide
          </button>
        </div>
      </div>

      <div class="template-band">
        <label class="template-field">
          <span>editable starter prompt using SA3 tags and Underfit's example values</span>
          <textarea rows="3" bind:value={templateText}></textarea>
        </label>
        <label class:disabled={!sharedPrompt.trim()} class="check-row">
          <input type="checkbox" bind:checked={includeSharedPrompt} disabled={!sharedPrompt.trim()} />
          <span>prepend shared phrase{sharedPrompt.trim() ? `: ${sharedPrompt.trim()}` : ""}</span>
        </label>
        <div class="template-actions">
          <button type="button" onclick={fillMissing} disabled={loading || !entries.length}>fill missing</button>
          <span>{captionedCount} of {entries.length} tracks have prompt text</span>
        </div>
      </div>

      {#if error}
        <div class="error-note">{error}</div>
      {/if}
      {#if message}
        <div class="success-note">{message}</div>
      {/if}

      {#if loading}
        <div class="empty">Scanning audio files...</div>
      {:else if !entries.length}
        <div class="empty">No supported audio files found in this folder.</div>
      {:else}
        <div class="editor">
          <div class="track-list">
            {#each entries as entry, index}
              <button
                type="button"
                class:active={index === selectedIndex}
                class="track-row"
                onclick={() => selectTrack(index)}
              >
                <span class="track-name">{entry.relativePath}</span>
                <span class:filled={!!entry.content.trim()} class="track-state">
                  {entry.content.trim() ? "txt" : "none"}
                </span>
              </button>
            {/each}
          </div>

          {#if selectedEntry}
            <div class="track-editor">
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
                <textarea rows="4" bind:value={selectedEntry.content} placeholder="Leave blank to train without a per-track prompt."></textarea>
              </label>
              <div class="dice-preview">
                <span>dice button result</span>
                <div>{selectedDicePrompt || "not added to the LoRA prompt pool"}</div>
                {#if selectedEntry.content.trim() !== selectedDicePrompt}
                  <small>A trailing BPM tag is omitted because Gary supplies tempo separately.</small>
                {/if}
              </div>

              <div class="track-actions">
                <button type="button" onclick={clearCurrent}>clear</button>
                <button type="button" onclick={restoreCurrent} disabled={selectedEntry.content === selectedEntry.originalContent}>restore</button>
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
        <span>{dirtyCount ? `${dirtyCount} unsaved change${dirtyCount === 1 ? "" : "s"}` : "all changes saved"}</span>
        <button type="button" class="accent" onclick={saveSidecars} disabled={saving || !dirtyCount}>
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

  .references {
    display: flex;
    gap: 12px;
    margin-top: 8px;
    flex-wrap: wrap;
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

  .check-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-primary);
  }

  .check-row.disabled {
    color: var(--text-muted);
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
    min-height: 340px;
    overflow: hidden;
  }

  .track-list {
    overflow: auto;
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

  .track-editor {
    min-width: 0;
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
    justify-content: flex-end;
    margin-top: 10px;
    flex-wrap: wrap;
  }

  .warning,
  .error-note,
  .success-note,
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

  .success-note {
    color: #9bd8aa;
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

  @media (max-width: 700px) {
    .sidecar-overlay {
      padding: 8px;
    }

    .editor {
      grid-template-columns: 1fr;
      overflow: auto;
    }

    .track-list {
      max-height: 180px;
      border-right: none;
      border-bottom: 1px solid var(--border);
    }

    .header,
    .footer,
    .template-actions {
      align-items: flex-start;
      flex-wrap: wrap;
    }
  }
</style>
