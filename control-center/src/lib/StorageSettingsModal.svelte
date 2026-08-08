<script lang="ts">
  import { open as openDialog } from "@tauri-apps/plugin-dialog";

  interface RuntimeStorageInfo {
    activeRoot: string;
    configuredRoot: string | null;
    defaultRoot: string;
    legacyRoot: string;
    configPath: string;
    pendingRestart: boolean;
    usingLegacyDefault: boolean;
  }

  let {
    open,
    info,
    busy = false,
    error,
    onChoose,
    onReset,
    onReveal,
    onClose,
  }: {
    open: boolean;
    info: RuntimeStorageInfo | null;
    busy?: boolean;
    error: string | null;
    onChoose: (path: string) => void;
    onReset: () => void;
    onReveal: (path: string) => void;
    onClose: () => void;
  } = $props();

  async function chooseFolder() {
    if (busy) return;
    const selected = await openDialog({
      directory: true,
      multiple: false,
      defaultPath: info?.activeRoot,
    });
    if (typeof selected === "string" && selected.trim()) {
      onChoose(selected);
    }
  }
</script>

{#if open}
  <div class="overlay">
    <button type="button" class="backdrop" aria-label="close storage settings" onclick={onClose}></button>
    <div
      class="modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="storage-modal-title"
      tabindex="-1"
    >
      <div class="eyebrow">storage</div>
      <div class="title-row">
        <div class="title" id="storage-modal-title">runtime storage</div>
        {#if info?.pendingRestart}
          <span class="pill">restart required</span>
        {/if}
      </div>

      {#if info}
        <div class="path-group">
          <div class="path-label">active</div>
          <button type="button" class="path-value" onclick={() => onReveal(info.activeRoot)} title="Open active storage folder">
            {info.activeRoot}
          </button>
        </div>

        {#if info.configuredRoot && info.configuredRoot !== info.activeRoot}
          <div class="path-group pending">
            <div class="path-label">next restart</div>
            <button type="button" class="path-value" onclick={() => onReveal(info.configuredRoot!)} title="Open configured storage folder">
              {info.configuredRoot}
            </button>
          </div>
        {/if}

        <div class="path-grid">
          <div>
            <div class="path-label">new install default</div>
            <div class="small-path">{info.defaultRoot}</div>
          </div>
          <div>
            <div class="path-label">legacy appdata</div>
            <div class="small-path">{info.legacyRoot}</div>
          </div>
        </div>

        {#if info.usingLegacyDefault}
          <div class="note">using legacy storage because existing data was found there.</div>
        {/if}
      {:else}
        <div class="body">loading storage paths...</div>
      {/if}

      {#if error}
        <div class="error-note">{error}</div>
      {/if}

      <div class="actions">
        <button type="button" onclick={chooseFolder} disabled={busy}>choose folder</button>
        <button type="button" onclick={onReset} disabled={busy}>clear custom</button>
        {#if info}
          <button type="button" onclick={() => onReveal(info.activeRoot)} disabled={busy}>open active</button>
        {/if}
        <button type="button" class="accent" onclick={onClose} disabled={busy}>close</button>
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
    width: min(640px, 100%);
    border: 1px solid var(--border);
    background: var(--bg-secondary);
    box-shadow: 0 22px 64px rgba(0, 0, 0, 0.5);
    padding: 20px;
  }

  .eyebrow {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-secondary);
  }

  .title-row {
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }

  .title {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .pill {
    border: 1px solid rgba(245, 185, 85, 0.45);
    color: #f5c46f;
    background: rgba(245, 185, 85, 0.1);
    font-size: 10px;
    line-height: 1;
    padding: 5px 7px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
  }

  .path-group {
    margin-top: 16px;
  }

  .path-group.pending {
    border-left: 2px solid #f5c46f;
    padding-left: 10px;
  }

  .path-label {
    margin-bottom: 6px;
    color: var(--text-secondary);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
  }

  .path-value,
  .small-path {
    width: 100%;
    min-width: 0;
    border: 1px solid var(--border);
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 1.45;
    text-align: left;
    padding: 9px 10px;
    overflow-wrap: anywhere;
  }

  .path-value {
    cursor: pointer;
  }

  .path-value:hover {
    border-color: var(--accent);
  }

  .path-grid {
    margin-top: 14px;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
  }

  .note,
  .body,
  .error-note {
    margin-top: 12px;
    font-size: 12px;
    line-height: 1.45;
  }

  .note,
  .body {
    color: var(--text-secondary);
  }

  .error-note {
    color: #ff8b8b;
  }

  .actions {
    margin-top: 18px;
    display: flex;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: 8px;
  }

  @media (max-width: 640px) {
    .path-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
