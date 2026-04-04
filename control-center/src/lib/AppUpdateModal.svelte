<script lang="ts">
  interface AppUpdateCheck {
    currentVersion: string;
    manifestUrl: string;
    checkedAtEpochMs: number;
    channel: string;
    latestVersion: string;
    updateAvailable: boolean;
    shouldPrompt: boolean;
    releaseNotesUrl: string | null;
    downloadUrl: string | null;
    sha256: string | null;
    publishedAt: string | null;
    notes: string[];
  }

  let {
    open,
    result,
    error,
    autoCheckEnabled,
    isSkipped,
    busy = false,
    onClose,
    onDownload,
    onViewReleaseNotes,
    onSkipVersion,
    onResumeReminders,
    onAutoCheckChange,
  }: {
    open: boolean;
    result: AppUpdateCheck | null;
    error: string | null;
    autoCheckEnabled: boolean;
    isSkipped: boolean;
    busy?: boolean;
    onClose: () => void;
    onDownload: () => void;
    onViewReleaseNotes: () => void;
    onSkipVersion: () => void;
    onResumeReminders: () => void;
    onAutoCheckChange: (value: boolean) => void;
  } = $props();

  function formatPublishedAt(value: string | null): string | null {
    if (!value) return null;
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return parsed.toLocaleString();
  }

  function shortHash(value: string | null): string | null {
    if (!value) return null;
    if (value.length <= 16) return value;
    return `${value.slice(0, 12)}...${value.slice(-8)}`;
  }
</script>

{#if open}
  <div class="overlay">
    <button type="button" class="backdrop" aria-label="close update prompt" onclick={onClose}></button>
    <div
      class="modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="update-modal-title"
      tabindex="-1"
    >
      <div class="eyebrow">gary4local updater</div>
      <div class="title" id="update-modal-title">
        {#if error}
          update check failed
        {:else if result?.updateAvailable}
          version {result.latestVersion} is ready
        {:else}
          you're up to date
        {/if}
      </div>

      {#if error}
        <div class="body">{error}</div>
        <div class="note">
          the app will keep working normally even if the update manifest cannot be reached.
        </div>
      {:else if result?.updateAvailable}
        <div class="body version-row">
          <span>installed: v{result.currentVersion}</span>
          <span class="arrow">-&gt;</span>
          <span>latest: v{result.latestVersion}</span>
        </div>

        {#if formatPublishedAt(result.publishedAt)}
          <div class="note">published {formatPublishedAt(result.publishedAt)}</div>
        {/if}

        {#if result.notes.length > 0}
          <div class="section-label">what's new</div>
          <ul class="notes">
            {#each result.notes as note}
              <li>{note}</li>
            {/each}
          </ul>
        {/if}

        {#if shortHash(result.sha256)}
          <div class="meta">sha256 {shortHash(result.sha256)}</div>
        {/if}
      {:else if result}
        <div class="body">gary4local v{result.currentVersion} is already the newest listed stable release.</div>
        <div class="note">manifest source: {result.manifestUrl}</div>
      {/if}

      <label class="auto-check">
        <input
          type="checkbox"
          checked={autoCheckEnabled}
          disabled={busy}
          onchange={(e) => onAutoCheckChange((e.currentTarget as HTMLInputElement).checked)}
        />
        <span>check for updates on startup</span>
      </label>

      <div class="actions">
        {#if result?.updateAvailable}
          <button onclick={onClose} disabled={busy}>not now</button>
          {#if isSkipped}
            <button onclick={onResumeReminders} disabled={busy}>resume reminders</button>
          {:else}
            <button onclick={onSkipVersion} disabled={busy}>skip this version</button>
          {/if}
          {#if result.releaseNotesUrl}
            <button onclick={onViewReleaseNotes} disabled={busy}>release notes</button>
          {/if}
          <button class="accent" onclick={onDownload} disabled={busy || !result.downloadUrl}>
            download update
          </button>
        {:else}
          {#if result?.releaseNotesUrl}
            <button onclick={onViewReleaseNotes} disabled={busy}>release notes</button>
          {/if}
          <button class="accent" onclick={onClose} disabled={busy}>close</button>
        {/if}
      </div>
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
    z-index: 60;
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
    width: min(520px, 100%);
    border: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(34, 34, 34, 0.98), rgba(20, 20, 20, 0.98));
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
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .body {
    margin-top: 12px;
    font-size: 13px;
    color: var(--text-primary);
    line-height: 1.55;
  }

  .version-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }

  .arrow {
    color: var(--accent);
    font-weight: 700;
  }

  .note {
    margin-top: 8px;
    font-size: 11px;
    color: var(--text-secondary);
    line-height: 1.45;
    word-break: break-word;
  }

  .section-label {
    margin-top: 16px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary);
    font-weight: 600;
  }

  .notes {
    margin-top: 8px;
    padding-left: 18px;
    color: var(--text-primary);
    line-height: 1.55;
  }

  .notes li + li {
    margin-top: 6px;
  }

  .meta {
    margin-top: 12px;
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  .auto-check {
    margin-top: 18px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--text-primary);
  }

  .auto-check input {
    accent-color: var(--accent);
  }

  .actions {
    margin-top: 20px;
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    flex-wrap: wrap;
  }
</style>
