<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";

  interface CareyAceSidecarFields {
    caption: string;
    genre: string;
    bpm: string;
    bpmSource: string;
    lmBpm: string;
    localBpm: string;
    filenameBpm: string;
    keyscale: string;
    keySource: string;
    lmKeyscale: string;
    localKeyscale: string;
    timesignature: string;
    language: string;
    isInstrumental: boolean;
    customTag: string;
    lyrics: string;
  }

  interface CareyAceDatasetSidecarEntry {
    audioPath: string;
    relativePath: string;
    sidecarPath: string;
    exists: boolean;
    rawContent: string;
    fields: CareyAceSidecarFields;
  }

  interface DraftEntry extends CareyAceDatasetSidecarEntry {
    originalFields: string;
  }

  interface CareyAceDatasetSidecarSaveResult {
    saved: number;
    removed: number;
    entries: CareyAceDatasetSidecarEntry[];
  }

  let {
    open,
    datasetPath,
    sharedTrigger,
    defaultInstrumental,
    onClose,
  }: {
    open: boolean;
    datasetPath: string;
    sharedTrigger: string;
    defaultInstrumental: boolean;
    onClose: () => void;
  } = $props();

  let entries: DraftEntry[] = $state([]);
  let selectedIndex = $state(0);
  let loading = $state(false);
  let saving = $state(false);
  let error = $state<string | null>(null);
  let message = $state<string | null>(null);
  let loadedPath = $state("");
  let wasOpen = false;
  const lyricsTemplate = `[Intro - a few words describing instrumentation]

[Verse - 0:27 timestamps work here too]

lyrics go here
even numbers
of syllables help, too

[Chorus]

you can do this (woo!)
if a background
singer goes (yeah!)

[Outro - description]`;

  function describeError(value: unknown): string {
    return value instanceof Error ? value.message : String(value);
  }

  function signature(fields: CareyAceSidecarFields): string {
    return JSON.stringify(fields);
  }

  function sourceLabel(value: string): string {
    return value.trim().replace(/_/g, " ");
  }

  function bpmProvenance(fields: CareyAceSidecarFields): string {
    const parts = [];
    if (fields.bpmSource.trim()) parts.push(`selected by ${sourceLabel(fields.bpmSource)}`);
    if (fields.lmBpm.trim()) parts.push(`LM ${fields.lmBpm.trim()}`);
    if (fields.localBpm.trim()) parts.push(`local ${fields.localBpm.trim()}`);
    if (fields.filenameBpm.trim()) parts.push(`filename ${fields.filenameBpm.trim()}`);
    return parts.join(" · ");
  }

  function keyProvenance(fields: CareyAceSidecarFields): string {
    const parts = [];
    if (fields.keySource.trim()) parts.push(`selected by ${sourceLabel(fields.keySource)}`);
    if (fields.lmKeyscale.trim()) parts.push(`LM ${fields.lmKeyscale.trim()}`);
    if (fields.localKeyscale.trim()) parts.push(`local ${fields.localKeyscale.trim()}`);
    return parts.join(" · ");
  }

  function toDrafts(items: CareyAceDatasetSidecarEntry[]): DraftEntry[] {
    return items.map((item) => ({
      ...item,
      originalFields: signature(item.fields),
    }));
  }

  async function loadSidecars() {
    if (!datasetPath.trim()) return;
    loading = true;
    error = null;
    message = null;
    try {
      const result = await invoke<CareyAceDatasetSidecarEntry[]>(
        "get_carey_ace_dataset_sidecars",
        { datasetPath }
      );
      entries = toDrafts(result);
      selectedIndex = Math.min(selectedIndex, Math.max(0, entries.length - 1));
      loadedPath = datasetPath;
    } catch (e) {
      error = describeError(e);
    } finally {
      loading = false;
    }
  }

  function replaceAllLyricsWithInstrumental() {
    entries = entries.map((entry) => ({
      ...entry,
      fields: {
        ...entry.fields,
        isInstrumental: true,
        lyrics: "[Instrumental]",
      },
    }));
    error = null;
    message = "Replaced every lyrics draft with [Instrumental]. Review the changes, then save when ready.";
  }

  function clearCurrent() {
    const entry = entries[selectedIndex];
    if (!entry) return;
    entries[selectedIndex] = {
      ...entry,
        fields: {
          caption: "",
          genre: "",
          bpm: "",
          bpmSource: "",
          lmBpm: "",
          localBpm: "",
          filenameBpm: "",
          keyscale: "",
          keySource: "",
          lmKeyscale: "",
          localKeyscale: "",
          timesignature: "",
          language: "",
        isInstrumental: false,
        customTag: "",
        lyrics: "",
      },
    };
    message = null;
  }

  function restoreCurrent() {
    const entry = entries[selectedIndex];
    if (!entry) return;
    entries[selectedIndex] = {
      ...entry,
      fields: JSON.parse(entry.originalFields),
    };
    message = null;
  }

  async function saveSidecars() {
    const changed = entries.filter((entry) => signature(entry.fields) !== entry.originalFields);
    if (!changed.length) {
      message = "No sidecar changes to save.";
      return;
    }

    saving = true;
    error = null;
    message = null;
    try {
      const result = await invoke<CareyAceDatasetSidecarSaveResult>(
        "save_carey_ace_dataset_sidecars",
        {
          datasetPath,
          sidecars: changed.map((entry) => ({
            audioPath: entry.audioPath,
            fields: entry.fields,
          })),
        }
      );
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
    entries.filter((entry) => signature(entry.fields) !== entry.originalFields).length
  );
  let captionedCount = $derived(entries.filter((entry) => entry.fields.caption.trim()).length);
  let metadataCount = $derived(entries.filter((entry) => entry.exists).length);

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
    <button type="button" class="sidecar-backdrop" aria-label="close ace-step sidecar editor" onclick={onClose}></button>
    <div class="sidecar-modal" role="dialog" aria-modal="true" aria-labelledby="ace-sidecar-title">
      <div class="header">
        <div>
          <div class="eyebrow">ace-step metadata</div>
          <div class="title" id="ace-sidecar-title">edit training sidecars</div>
        </div>
        <div class="header-actions">
          <button type="button" onclick={loadSidecars} disabled={loading || saving}>refresh</button>
          <button type="button" onclick={onClose}>close</button>
        </div>
      </div>

      <div class="body">
        Review each track’s caption and metadata. Your edits are saved to the matching `.txt` file beside the audio.
      </div>

      <div class="template-band">
        <div class="template-actions">
          <button type="button" onclick={replaceAllLyricsWithInstrumental} disabled={loading || !entries.length}>
            replace all lyrics with [Instrumental]
          </button>
          <small>optional cleanup; this overwrites generated lyric descriptions only after you save.</small>
        </div>
        <span>{captionedCount} captions, {metadataCount} sidecars, {entries.length} tracks</span>
        {#if defaultInstrumental}
          <small>training is set to instrumental; individual lyric drafts remain your choice.</small>
        {/if}
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
                <span class:filled={!!entry.fields.caption.trim()} class="track-state">
                  {entry.fields.caption.trim() ? "caption" : entry.exists ? "txt" : "none"}
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

              <label class="prompt-field">
                <span>caption</span>
                <textarea rows="4" bind:value={selectedEntry.fields.caption} placeholder="What does this track sound like?"></textarea>
              </label>

              <div class="field-grid">
                <label class="field">
                  <span>genre</span>
                  <input type="text" bind:value={selectedEntry.fields.genre} />
                </label>
                <label class="field">
                  <span>bpm</span>
                  <input type="text" inputmode="decimal" bind:value={selectedEntry.fields.bpm} />
                  {#if bpmProvenance(selectedEntry.fields)}
                    <small class="provenance">{bpmProvenance(selectedEntry.fields)}</small>
                  {/if}
                </label>
                <label class="field">
                  <span>key</span>
                  <input type="text" bind:value={selectedEntry.fields.keyscale} placeholder="D minor" />
                  {#if keyProvenance(selectedEntry.fields)}
                    <small class="provenance">{keyProvenance(selectedEntry.fields)}</small>
                  {/if}
                </label>
                <label class="field">
                  <span>time signature</span>
                  <input type="text" bind:value={selectedEntry.fields.timesignature} placeholder="optional" />
                </label>
                <label class="field">
                  <span>language</span>
                  <input type="text" bind:value={selectedEntry.fields.language} placeholder="unknown" />
                </label>
                <label class="field">
                  <span>per-track tag override</span>
                  <input type="text" bind:value={selectedEntry.fields.customTag} placeholder={sharedTrigger || "optional trigger"} />
                  <small class="provenance">used only when the trainer’s shared trigger tag is blank.</small>
                </label>
              </div>

              <label class="check-row">
                <input type="checkbox" bind:checked={selectedEntry.fields.isInstrumental} />
                <span>instrumental</span>
              </label>

              <label class="prompt-field">
                <span>lyrics</span>
                <textarea rows="6" bind:value={selectedEntry.fields.lyrics} placeholder={lyricsTemplate}></textarea>
                <small class="provenance">lyrics are BYOL; this grey template is only a guide and is not saved unless you type it.</small>
              </label>

              <div class="track-actions">
                <button type="button" onclick={clearCurrent}>clear</button>
                <button type="button" onclick={restoreCurrent} disabled={signature(selectedEntry.fields) === selectedEntry.originalFields}>restore</button>
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
    z-index: 78;
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
    width: min(980px, 100%);
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

  .header-actions,
  .template-actions {
    flex-wrap: wrap;
  }

  .eyebrow,
  .prompt-field span,
  .field span {
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

  .template-band {
    display: grid;
    gap: 9px;
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
  }

  .template-band span,
  .template-band small,
  .footer span,
  .track-heading > span {
    color: var(--text-secondary);
    font-size: 11px;
  }

  .editor {
    display: grid;
    grid-template-columns: minmax(220px, 0.34fr) minmax(0, 1fr);
    min-height: 430px;
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

  .prompt-field,
  .field {
    display: grid;
    gap: 6px;
  }

  .prompt-field {
    margin-top: 14px;
  }

  .field-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 12px;
  }

  input,
  textarea {
    width: 100%;
    border: 1px solid var(--border);
    background: var(--bg-primary);
    color: var(--text-primary);
    font: 12px/1.5 var(--font-mono);
    padding: 8px;
    user-select: text;
    -webkit-user-select: text;
  }

  textarea {
    resize: vertical;
  }

  textarea::placeholder {
    color: var(--text-muted);
    opacity: 0.74;
  }

  .provenance {
    color: var(--text-muted);
    font: 10px/1.4 var(--font-mono);
  }

  .check-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 12px;
    font-size: 11px;
    color: var(--text-primary);
  }

  .check-row input {
    width: 15px;
    height: 15px;
    accent-color: var(--accent);
  }

  .track-actions {
    justify-content: flex-end;
    margin-top: 10px;
    flex-wrap: wrap;
  }

  .error-note,
  .success-note,
  .empty {
    margin: 10px 18px 0;
    font-size: 11px;
    line-height: 1.45;
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

  .accent {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }

  @media (max-width: 760px) {
    .sidecar-overlay {
      padding: 8px;
    }

    .editor,
    .field-grid {
      grid-template-columns: 1fr;
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
