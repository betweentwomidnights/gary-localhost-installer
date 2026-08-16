<script lang="ts">
  import { open as openDialog } from "@tauri-apps/plugin-dialog";

  interface RuntimeStorageInfo {
    activeRoot: string;
    configuredRoot: string | null;
    startupRoot: string;
    defaultRoot: string;
    legacyRoot: string;
    configPath: string;
    pendingRestart: boolean;
    usingLegacyDefault: boolean;
    defaultRootIsLegacy: boolean;
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
    pendingRoot: string;
    legacyRoot: string;
    defaultHfCacheRoot: string;
    cleanupItems: LegacyStorageCleanupItem[];
    loraCandidates: LegacyLoraMigrationCandidate[];
    storageMoveLoraCandidates: LegacyLoraMigrationCandidate[];
    totalCleanupBytes: number;
    totalLoraBytes: number;
    totalStorageMoveLoraBytes: number;
    canCleanup: boolean;
    canMigrateLoras: boolean;
    canMigrateStorageLoras: boolean;
  }

  interface RuntimeCacheInfo {
    uvCachePath: string;
    uvCacheBytes: number;
  }

  interface ServiceEnvInfo {
    serviceId: string;
    displayName: string;
    envPath: string;
    envBytes: number;
    present: boolean;
    blockedReason: string | null;
  }

  let {
    open,
    info,
    busy = false,
    error,
    maintenanceInfo,
    maintenanceBusy = false,
    maintenanceError,
    maintenanceWarning,
    maintenanceMessage,
    cacheInfo,
    cacheBusy = false,
    cacheError,
    cacheMessage,
    serviceEnvs = [],
    serviceEnvBusy = null,
    serviceEnvError,
    serviceEnvMessage,
    onRemoveServiceEnv,
    restarting = false,
    onChoose,
    onReset,
    onReveal,
    onRefreshMaintenance,
    onMigrateLoras,
    onMigrateStorageLoras,
    onCleanupLegacy,
    onClearUvCache,
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
    maintenanceWarning: string | null;
    maintenanceMessage: string | null;
    cacheInfo: RuntimeCacheInfo | null;
    cacheBusy?: boolean;
    cacheError: string | null;
    cacheMessage: string | null;
    serviceEnvs?: ServiceEnvInfo[];
    serviceEnvBusy?: string | null;
    serviceEnvError: string | null;
    serviceEnvMessage: string | null;
    onRemoveServiceEnv: (serviceId: string) => void;
    restarting?: boolean;
    onChoose: (path: string) => void;
    onReset: () => void;
    onReveal: (path: string) => void;
    onRefreshMaintenance: () => void;
    onMigrateLoras: () => void;
    onMigrateStorageLoras: () => void;
    onCleanupLegacy: () => void;
    onClearUvCache: () => void;
    onRestart: () => void;
    onClose: () => void;
  } = $props();

  let loraListExpanded = $state(false);
  let storageMoveListExpanded = $state(false);
  let cleanupListExpanded = $state(false);

  let visibleLoraCandidates = $derived(
    maintenanceInfo
      ? loraListExpanded
        ? maintenanceInfo.loraCandidates
        : maintenanceInfo.loraCandidates.slice(0, 4)
      : [],
  );
  let visibleCleanupItems = $derived(
    maintenanceInfo
      ? cleanupListExpanded
        ? maintenanceInfo.cleanupItems
        : maintenanceInfo.cleanupItems.slice(0, 5)
      : [],
  );
  let visibleStorageMoveLoras = $derived(
    maintenanceInfo
      ? storageMoveListExpanded
        ? maintenanceInfo.storageMoveLoraCandidates
        : maintenanceInfo.storageMoveLoraCandidates.slice(0, 4)
      : [],
  );
  let hiddenLoraCount = $derived(
    maintenanceInfo
      ? Math.max(maintenanceInfo.loraCandidates.length - visibleLoraCandidates.length, 0)
      : 0,
  );
  let hiddenCleanupCount = $derived(
    maintenanceInfo
      ? Math.max(maintenanceInfo.cleanupItems.length - visibleCleanupItems.length, 0)
      : 0,
  );
  let hiddenStorageMoveLoraCount = $derived(
    maintenanceInfo
      ? Math.max(
          maintenanceInfo.storageMoveLoraCandidates.length - visibleStorageMoveLoras.length,
          0,
        )
      : 0,
  );
  let storageMovePending = $derived(
    !!info && info.pendingRestart && info.startupRoot !== info.activeRoot,
  );
  function normalizePath(path: string | null | undefined) {
    return (path ?? "").replaceAll("/", "\\").replace(/\\+$/, "").toLowerCase();
  }
  let defaultPathVisibleElsewhere = $derived(
    !!info &&
      (normalizePath(info.defaultRoot) === normalizePath(info.activeRoot) ||
        (info.pendingRestart &&
          normalizePath(info.defaultRoot) === normalizePath(info.startupRoot))),
  );
  let showDefaultPath = $derived(!!info && !defaultPathVisibleElsewhere);
  let defaultPathLabel = $derived(
    info?.configuredRoot ? "default if custom is removed" : "default",
  );
  let defaultStorageNote = $derived(
    info?.defaultRootIsLegacy
      ? info?.configuredRoot || info?.pendingRestart
        ? "Gary's default on this machine is legacy AppData because existing data was found there."
        : "using default storage: legacy AppData because existing data was found there."
      : info?.configuredRoot || info?.pendingRestart
        ? "Gary's default on this machine is the gary4local-data folder beside the app."
        : "using default storage beside the app.",
  );
  let resetButtonLabel = $derived(
    info?.configuredRoot
      ? "use default"
      : info?.pendingRestart
        ? "default pending"
        : "using default",
  );
  let resetButtonTitle = $derived(
    info?.configuredRoot
      ? `Remove the custom storage setting. After restart, Gary uses ${info.defaultRoot}.`
      : "Gary is already using its default storage choice.",
  );

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

  function confirmClearUvCache() {
    const size = formatBytes(cacheInfo?.uvCacheBytes ?? 0);
    const confirmed = window.confirm(
      `Clear ${size} from the UV package cache?\n\n` +
      "Installed service environments and models are not removed. " +
      "UV will download packages again when environments are rebuilt."
    );
    if (confirmed) onClearUvCache();
  }

  const installedEnvs = $derived(serviceEnvs.filter((env) => env.present));
  const totalEnvBytes = $derived(
    installedEnvs.reduce((sum, env) => sum + env.envBytes, 0)
  );

  function confirmRemoveServiceEnv(env: ServiceEnvInfo) {
    const confirmed = window.confirm(
      `Remove ${env.displayName}'s environment and free ${formatBytes(env.envBytes)}?\n\n` +
      "Downloaded models are kept. This service can't run again until you rebuild " +
      "its environment, which re-downloads its packages."
    );
    if (confirmed) onRemoveServiceEnv(env.serviceId);
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

        {#if info.pendingRestart && info.startupRoot !== info.activeRoot}
          <div class="path-group pending">
            <div class="path-label">next restart</div>
            <button type="button" class="path-value" onclick={() => onReveal(info.startupRoot)} title="Open next restart storage folder">
              {info.startupRoot}
            </button>
          </div>
        {/if}

        {#if showDefaultPath}
          <div class="path-group">
            <div class="path-label">{defaultPathLabel}</div>
            <button type="button" class="path-value" onclick={() => onReveal(info.defaultRoot)} title="Open default storage folder">
              {info.defaultRoot}
            </button>
          </div>
        {/if}

        <div class="note">{defaultStorageNote}</div>
      {:else}
        <div class="body">loading storage paths...</div>
      {/if}

      {#if error}
        <div class="error-note">{error}</div>
      {/if}

      <div class="active-cache">
        <div class="section-row">
          <div>
            <div class="section-title">UV package cache</div>
            <div class="section-copy">
              {cacheInfo ? formatBytes(cacheInfo.uvCacheBytes) : "scanning..."}
            </div>
          </div>
          <button
            type="button"
            class="small-action"
            onclick={confirmClearUvCache}
            disabled={busy || cacheBusy || !cacheInfo || cacheInfo.uvCacheBytes === 0}
          >{cacheBusy ? "clearing..." : "clear cache"}</button>
        </div>
        {#if cacheInfo}
          <button type="button" class="cache-path" onclick={() => onReveal(cacheInfo.uvCachePath)} title="Open UV cache folder">
            {cacheInfo.uvCachePath}
          </button>
        {/if}
        <div class="note">UV keeps downloaded Python packages here to make environment rebuilds faster. Clearing it does not remove installed environments or models; future rebuilds download those packages again.</div>
        {#if cacheMessage}<div class="success-note">{cacheMessage}</div>{/if}
        {#if cacheError}<div class="error-note">{cacheError}</div>{/if}
      </div>

      <div class="active-cache">
        <div class="section-row">
          <div>
            <div class="section-title">service environments</div>
            <div class="section-copy">
              {installedEnvs.length > 0
                ? `${installedEnvs.length} installed - ${formatBytes(totalEnvBytes)}`
                : "none installed"}
            </div>
          </div>
        </div>
        {#each serviceEnvs as env (env.serviceId)}
          <div class="env-row">
            <div class="env-label">
              <span class="env-name">{env.displayName}</span>
              <span class="env-size">{env.present ? formatBytes(env.envBytes) : "not installed"}</span>
            </div>
            <button
              type="button"
              class="small-action"
              title={env.blockedReason ?? `Remove ${env.displayName}'s environment`}
              onclick={() => confirmRemoveServiceEnv(env)}
              disabled={busy || serviceEnvBusy !== null || env.blockedReason !== null}
            >{serviceEnvBusy === env.serviceId ? "removing..." : "remove env"}</button>
          </div>
          {#if env.blockedReason && env.present}
            <div class="env-blocked">{env.blockedReason}</div>
          {/if}
        {/each}
        <div class="note">Each environment is the Python install for one model, several GB apiece. Removing one keeps that model's downloaded weights and frees the packages; rebuild it from the service when you want to run it again.</div>
        {#if serviceEnvMessage}<div class="success-note">{serviceEnvMessage}</div>{/if}
        {#if serviceEnvError}<div class="error-note">{serviceEnvError}</div>{/if}
      </div>

      <div class="maintenance">
        <div class="section-row">
          <div>
            <div class="section-title">old envs, models, and caches</div>
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
          <div class="note">cleanup unlocks after you choose another storage folder and restart the app.</div>
        {/if}

        {#if maintenanceInfo?.canMigrateStorageLoras}
          <div class="maintenance-row warning">
            <div>
              <div class="row-title">LoRAs in current storage</div>
              <div class="row-copy">
                {maintenanceInfo.storageMoveLoraCandidates.length} LoRA{maintenanceInfo.storageMoveLoraCandidates.length === 1 ? "" : "s"}
                can be copied before restart - {formatBytes(maintenanceInfo.totalStorageMoveLoraBytes)}
              </div>
            </div>
            <button type="button" onclick={onMigrateStorageLoras} disabled={busy || maintenanceBusy}>copy to next storage</button>
          </div>
          <div class="item-list">
            {#each visibleStorageMoveLoras as lora}
              <div class="item-row">
                <span>{lora.service} - {lora.name}</span>
                <span>{formatBytes(lora.bytes)}</span>
              </div>
            {/each}
            {#if maintenanceInfo.storageMoveLoraCandidates.length > 4}
              <button
                type="button"
                class="item-row expand-row"
                onclick={() => (storageMoveListExpanded = !storageMoveListExpanded)}
              >
                <span>{storageMoveListExpanded ? "show fewer" : `${hiddenStorageMoveLoraCount} more`}</span>
                <span>{storageMoveListExpanded ? "-" : "+"}</span>
              </button>
            {/if}
          </div>
          <div class="note">This copies LoRAs into the folder used after restart. It does not delete them from the current storage.</div>
        {:else if storageMovePending && maintenanceInfo}
          <div class="note">no Gary-managed LoRAs need copying to next storage. LoRAs already registered there stay available after restart.</div>
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
            <button type="button" onclick={onMigrateLoras} disabled={busy || maintenanceBusy}>copy LoRAs</button>
          </div>
          <div class="item-list">
            {#each visibleLoraCandidates as lora}
              <div class="item-row">
                <span>{lora.service} - {lora.name}</span>
                <span>{formatBytes(lora.bytes)}</span>
              </div>
            {/each}
            {#if maintenanceInfo.loraCandidates.length > 4}
              <button
                type="button"
                class="item-row expand-row"
                onclick={() => (loraListExpanded = !loraListExpanded)}
              >
                <span>{loraListExpanded ? "show fewer" : `${hiddenLoraCount} more`}</span>
                <span>{loraListExpanded ? "-" : "+"}</span>
              </button>
            {/if}
          </div>
        {/if}

        {#if maintenanceInfo?.canCleanup}
          <div class="maintenance-row">
            <div>
              <div class="row-title">old envs, models, and caches</div>
              <div class="row-copy">
                {maintenanceInfo.cleanupItems.length} item{maintenanceInfo.cleanupItems.length === 1 ? "" : "s"}
                - {formatBytes(maintenanceInfo.totalCleanupBytes)}
              </div>
            </div>
            <button type="button" onclick={onCleanupLegacy} disabled={busy || maintenanceBusy}>clean up listed items</button>
          </div>
          <div class="item-list">
            {#each visibleCleanupItems as item}
              <button type="button" class="item-row path-row" onclick={() => onReveal(item.path)} title="Open item location">
                <span>{item.label}</span>
                <span>{formatBytes(item.bytes)}</span>
              </button>
            {/each}
            {#if maintenanceInfo.cleanupItems.length > 5}
              <button
                type="button"
                class="item-row expand-row"
                onclick={() => (cleanupListExpanded = !cleanupListExpanded)}
              >
                <span>{cleanupListExpanded ? "show fewer" : `${hiddenCleanupCount} more`}</span>
                <span>{cleanupListExpanded ? "-" : "+"}</span>
              </button>
            {/if}
          </div>
          <div class="note">Only the listed items are deleted. Copied LoRAs, settings, and tokens stay in old storage.</div>
        {:else if maintenanceInfo && maintenanceInfo.activeRoot !== maintenanceInfo.legacyRoot && !maintenanceInfo.canMigrateLoras}
          <div class="note">no old model or environment cleanup found.</div>
        {/if}
      </div>

      {#if maintenanceMessage}
        <div class="note">{maintenanceMessage}</div>
      {/if}

      {#if maintenanceWarning}
        <div class="warning-note multiline">{maintenanceWarning}</div>
      {/if}

      {#if maintenanceError}
        <div class="error-note multiline">{maintenanceError}</div>
      {/if}

      <div class="actions">
        <button type="button" onclick={chooseFolder} disabled={busy}>choose folder</button>
        <button
          type="button"
          onclick={onReset}
          disabled={busy || !info?.configuredRoot}
          title={resetButtonTitle}
        >
          {resetButtonLabel}
        </button>
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

  .path-value {
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

  .warning-note {
    margin-top: 12px;
    color: #f5c46f;
    font-size: 12px;
    line-height: 1.45;
  }

  .error-note.multiline,
  .warning-note.multiline {
    white-space: pre-wrap;
  }

  .maintenance {
    margin-top: 16px;
    border-top: 1px solid var(--border);
    padding-top: 14px;
  }

  .active-cache {
    margin-top: 16px;
    border-top: 1px solid var(--border);
    padding-top: 14px;
  }

  .env-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-top: 8px;
  }

  .env-label {
    display: flex;
    align-items: baseline;
    gap: 8px;
    min-width: 0;
  }

  .env-name {
    font-size: 12px;
    overflow-wrap: anywhere;
  }

  .env-size {
    color: var(--text-muted);
    font-size: 11px;
    font-family: var(--font-mono);
  }

  .env-blocked {
    color: var(--text-muted);
    font-size: 10px;
    margin-top: 2px;
  }

  .cache-path {
    width: 100%;
    margin-top: 8px;
    border: none;
    background: transparent;
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 10px;
    text-align: left;
    padding: 0;
    overflow-wrap: anywhere;
    cursor: pointer;
  }

  .cache-path:hover {
    color: var(--text-primary);
    text-decoration: underline;
  }

  .success-note {
    margin-top: 10px;
    color: #9bd8aa;
    font-size: 12px;
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

  .path-row,
  .expand-row {
    cursor: pointer;
  }

  .path-row:hover,
  .expand-row:hover {
    color: var(--text-primary);
  }

  .expand-row {
    color: var(--accent);
  }

  .actions {
    margin-top: 18px;
    display: flex;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: 8px;
  }

</style>
