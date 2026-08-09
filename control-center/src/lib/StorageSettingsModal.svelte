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

  interface LegacyStorageCleanupItem {
    id: string;
    label: string;
    path: string;
    bytes: number;
  }

  interface LegacyLoraMigrationCandidate {
    service: string;
    name: string;
    sourcePath: string;
    targetPath: string;
    bytes: number;
  }

  interface LegacyStorageMaintenanceInfo {
    activeRoot: string;
    legacyRoot: string;
    defaultHfCacheRoot: string;
    cleanupItems: LegacyStorageCleanupItem[];
    loraCandidates: LegacyLoraMigrationCandidate[];
    totalCleanupBytes: number;
    totalLoraBytes: number;
    canCleanup: boolean;
    canMigrateLoras: boolean;
  }

  let {
    open,
    info,
    busy = false,
    error,
    maintenanceInfo,
    maintenanceBusy = false,
    maintenanceError,
    maintenanceMessage,
    restarting = false,
    onChoose,
    onReset,
    onReveal,
    onRefreshMaintenance,
    onMigrateLoras,
    onCleanupLegacy,
    onRestart,
    onClose,
  }: {
    open: boolean;
    info: RuntimeStorageInfo | null;
    busy?: boolean;
    error: string | null;
    maintenanceInfo: LegacyStorageMaintenanceInfo | null;
    maintenanceBusy?: boolean;
    maintenanceError: string | null;
    maintenanceMessage: string | null;
    restarting?: boolean;
    onChoose: (path: string) => void;
    onReset: () => void;
    onReveal: (path: string) => void;
    onRefreshMaintenance: () => void;
    onMigrateLoras: () => void;
    onCleanupLegacy: () => void;
    onRestart: () => void;
    onClose: () => void;
  } = $props();

  function formatBytes(bytes: number) {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    const precision = value >= 10 || unit === 0 ? 0 : 1;
    return `${value.toFixed(precision)} ${units[unit]}`;
  }

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
          <button
            type="button"
            class="restart-action"
            onclick={onRestart}
            disabled={busy || maintenanceBusy || restarting}
          >
            {restarting ? "restarting..." : "restart app"}
          </button>
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

      <div class="maintenance">
        <div class="section-row">
          <div>
            <div class="section-title">old storage</div>
            <div class="section-copy">
              {#if maintenanceInfo}
                {maintenanceInfo.cleanupItems.length} cleanup item{maintenanceInfo.cleanupItems.length === 1 ? "" : "s"}
                - {formatBytes(maintenanceInfo.totalCleanupBytes)}
              {:else}
                scanning legacy locations...
              {/if}
            </div>
          </div>
          <button type="button" class="small-action" onclick={onRefreshMaintenance} disabled={busy || maintenanceBusy}>
            {maintenanceBusy ? "scanning..." : "refresh"}
          </button>
        </div>

        {#if maintenanceInfo && maintenanceInfo.activeRoot === maintenanceInfo.legacyRoot}
          <div class="note">cleanup unlocks after storage is moved off legacy AppData and the app restarts.</div>
        {/if}

        {#if maintenanceInfo?.canMigrateLoras}
          <div class="maintenance-row warning">
            <div>
              <div class="row-title">trained LoRAs found</div>
              <div class="row-copy">
                {maintenanceInfo.loraCandidates.length} registered LoRA{maintenanceInfo.loraCandidates.length === 1 ? "" : "s"}
                - {formatBytes(maintenanceInfo.totalLoraBytes)}
              </div>
            </div>
            <button type="button" onclick={onMigrateLoras} disabled={busy || maintenanceBusy}>migrate LoRAs</button>
          </div>
          <div class="item-list">
            {#each maintenanceInfo.loraCandidates.slice(0, 4) as lora}
              <div class="item-row">
                <span>{lora.service} - {lora.name}</span>
                <span>{formatBytes(lora.bytes)}</span>
              </div>
            {/each}
            {#if maintenanceInfo.loraCandidates.length > 4}
              <div class="item-row muted">
                <span>{maintenanceInfo.loraCandidates.length - 4} more</span>
                <span></span>
              </div>
            {/if}
          </div>
        {/if}

        {#if maintenanceInfo?.canCleanup}
          <div class="maintenance-row">
            <div>
              <div class="row-title">old envs and models</div>
              <div class="row-copy">
                {maintenanceInfo.cleanupItems.length} item{maintenanceInfo.cleanupItems.length === 1 ? "" : "s"}
                - {formatBytes(maintenanceInfo.totalCleanupBytes)}
              </div>
            </div>
            <button type="button" onclick={onCleanupLegacy} disabled={busy || maintenanceBusy}>clean up old storage</button>
          </div>
          <div class="item-list">
            {#each maintenanceInfo.cleanupItems.slice(0, 5) as item}
              <button type="button" class="item-row path-row" onclick={() => onReveal(item.path)} title="Open item location">
                <span>{item.label}</span>
                <span>{formatBytes(item.bytes)}</span>
              </button>
            {/each}
            {#if maintenanceInfo.cleanupItems.length > 5}
              <div class="item-row muted">
                <span>{maintenanceInfo.cleanupItems.length - 5} more</span>
                <span></span>
              </div>
            {/if}
          </div>
        {:else if maintenanceInfo && maintenanceInfo.activeRoot !== maintenanceInfo.legacyRoot && !maintenanceInfo.canMigrateLoras}
          <div class="note">no old model or environment cleanup found.</div>
        {/if}
      </div>

      {#if maintenanceMessage}
        <div class="note">{maintenanceMessage}</div>
      {/if}

      {#if maintenanceError}
        <div class="error-note multiline">{maintenanceError}</div>
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
    max-height: calc(100vh - 48px);
    overflow-y: auto;
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

  .restart-action {
    border-color: rgba(245, 185, 85, 0.6);
    color: #f5d08b;
    background: rgba(245, 185, 85, 0.12);
    min-width: 104px;
  }

  .restart-action:hover:not(:disabled) {
    border-color: #f5c46f;
    background: rgba(245, 185, 85, 0.18);
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

  .error-note.multiline {
    white-space: pre-wrap;
  }

  .maintenance {
    margin-top: 16px;
    border-top: 1px solid var(--border);
    padding-top: 14px;
  }

  .section-row,
  .maintenance-row,
  .item-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .section-title,
  .row-title {
    color: var(--text-primary);
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }

  .section-copy,
  .row-copy {
    margin-top: 4px;
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 1.35;
  }

  .small-action {
    min-width: 84px;
  }

  .maintenance-row {
    margin-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-top: 12px;
  }

  .maintenance-row.warning {
    border-top-color: rgba(245, 185, 85, 0.35);
  }

  .item-list {
    margin-top: 8px;
    border: 1px solid var(--border);
    background: var(--bg-primary);
  }

  .item-row {
    width: 100%;
    min-width: 0;
    border: none;
    border-bottom: 1px solid var(--border);
    background: transparent;
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.35;
    padding: 8px 10px;
    text-align: left;
  }

  .item-row:last-child {
    border-bottom: none;
  }

  .item-row span {
    min-width: 0;
    overflow-wrap: anywhere;
  }

  .item-row span:last-child {
    flex: 0 0 auto;
    color: var(--text-primary);
    font-family: var(--font-mono);
  }

  .item-row.muted {
    color: var(--text-secondary);
  }

  .path-row {
    cursor: pointer;
  }

  .path-row:hover {
    color: var(--text-primary);
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
