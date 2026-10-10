<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";

  interface BuildStatus {
    building: boolean;
    current_step: number;
    total_steps: number;
    step_label: string;
    log: string;
    error: string | null;
  }

  interface ServiceInfo {
    id: string;
    display_name: string;
    port: number;
    status: "stopped" | "starting" | "running" | "unhealthy" | "failed";
    pid: number | null;
    error: string | null;
    env_exists: boolean;
    build_status: BuildStatus | null;
    runtime: "python" | "native";
    native_backend: string | null;
    native_update_available: boolean;
    native_fallback_reason: string | null;
    start_blocker: string | null;
    sa3_migration_complete: boolean;
  }

  let {
    service,
    selected,
    onSelect,
    hasModels = false,
    onShowModels = () => {},
    hasCareyLoras = false,
    onManageCareyLoras = () => {},
    hasCareyAceTraining = false,
    onTrainCareyAce = () => {},
    hasSa3Loras = false,
    onManageSa3Loras = () => {},
    hasSa3LoraTraining = false,
    onTrainSa3Lora = () => {},
    hasSa3Migration = false,
    onMigrateSa3Runtime = () => {},
  }: {
    service: ServiceInfo;
    selected: boolean;
    onSelect: () => void;
    hasModels?: boolean;
    onShowModels?: () => void;
    hasCareyLoras?: boolean;
    onManageCareyLoras?: () => void;
    hasCareyAceTraining?: boolean;
    onTrainCareyAce?: () => void;
    hasSa3Loras?: boolean;
    onManageSa3Loras?: () => void;
    hasSa3LoraTraining?: boolean;
    onTrainSa3Lora?: () => void;
    hasSa3Migration?: boolean;
    onMigrateSa3Runtime?: () => void;
  } = $props();

  const statusColors: Record<string, string> = {
    running: "var(--green)",
    starting: "var(--yellow)",
    unhealthy: "var(--orange)",
    failed: "var(--red)",
    stopped: "var(--gray)",
  };

  function statusColor(s: string): string {
    return statusColors[s] || "var(--gray)";
  }

  let isBuilding = $derived(service.build_status?.building ?? false);
  // A native service downloads a prebuilt runtime instead of building a venv.
  let isNative = $derived(service.runtime === "native");
  let buildLabel = $derived(
    isNative
      ? !service.env_exists
        ? "install runtime"
        : service.native_update_available
          ? "update runtime"
          : "reinstall runtime"
      : service.env_exists
        ? "rebuild env"
        : "build env"
  );
  let buildProgress = $derived(
    service.build_status
      ? Math.round((service.build_status.current_step / service.build_status.total_steps) * 100)
      : 0
  );

  async function startService() {
    try { await invoke("start_service", { serviceId: service.id }); } catch (e) { console.error(e); }
  }
  async function stopService() {
    try { await invoke("stop_service", { serviceId: service.id }); } catch (e) { console.error(e); }
  }
  async function restartService() {
    try { await invoke("restart_service", { serviceId: service.id }); } catch (e) { console.error(e); }
  }
  async function rebuildEnv() {
    try { await invoke("rebuild_env", { serviceId: service.id }); } catch (e) { console.error(e); }
  }
</script>

<div
  class="service-row"
  class:selected
  onclick={onSelect}
  role="button"
  tabindex="0"
  onkeydown={(e) => e.key === "Enter" && onSelect()}
>
  <div class="row-top">
    <span class="status-dot" style="background: {statusColor(service.status)}"></span>
    <div class="info">
      <span class="name">{service.display_name}</span>
      <span class="meta">:{service.port} {#if isNative && service.native_backend}&middot; {service.native_backend} {/if}{#if service.pid}&middot; PID {service.pid}{/if}</span>
    </div>
  </div>

  {#if service.error}
    <div class="error">{service.error}</div>
  {/if}

  {#if isBuilding}
    <div class="build-progress">
      <div class="progress-bar">
        <div class="progress-fill" style="width: {buildProgress}%"></div>
      </div>
      <span class="build-label">{service.build_status?.step_label}</span>
    </div>
  {:else if service.build_status?.error}
    <div class="error">build failed: {service.build_status.error}</div>
  {:else if service.build_status && !service.build_status.building && service.build_status.current_step > 0}
    <div class="build-done">{isNative ? "runtime installed" : "build complete"}</div>
  {/if}

  {#if service.env_exists && service.start_blocker && (service.status === "stopped" || service.status === "failed")}
    <div class="blocker">{service.start_blocker}</div>
  {/if}

  <div class="controls">
    {#if service.status === "stopped" || service.status === "failed"}
      <button onclick={(e) => { e.stopPropagation(); startService(); }} disabled={!service.env_exists || isBuilding || !!service.start_blocker}>start</button>
    {:else}
      <button onclick={(e) => { e.stopPropagation(); stopService(); }}>stop</button>
      <button onclick={(e) => { e.stopPropagation(); restartService(); }}>restart</button>
    {/if}
    <button onclick={(e) => { e.stopPropagation(); rebuildEnv(); }} disabled={isBuilding}>
      {#if isBuilding}
        {isNative ? "installing..." : "building..."}
      {:else}
        {buildLabel}
      {/if}
    </button>
    {#if hasModels}
      <!-- Native models download without the runtime; Python ones need the env. -->
      <button class="models-btn" onclick={(e) => { e.stopPropagation(); onShowModels(); }} disabled={!service.env_exists && !isNative}>
        models
      </button>
    {/if}
    {#if hasCareyLoras}
      <button class="lora-btn" onclick={(e) => { e.stopPropagation(); onManageCareyLoras(); }}>
        add lora
      </button>
    {/if}
    {#if hasCareyAceTraining}
      <button class="lora-btn" onclick={(e) => { e.stopPropagation(); onTrainCareyAce(); }}>
        train lora
      </button>
    {/if}
    {#if hasSa3Loras}
      <button class="lora-btn" onclick={(e) => { e.stopPropagation(); onManageSa3Loras(); }}>
        add lora
      </button>
    {/if}
    {#if hasSa3LoraTraining || hasSa3Migration}<div class="runtime-actions">
    {#if hasSa3LoraTraining}
      <button class="lora-btn" onclick={(e) => { e.stopPropagation(); onTrainSa3Lora(); }}>
        train lora
      </button>
    {/if}
    {#if hasSa3Migration}
      <button class="migration-btn" onclick={(e) => { e.stopPropagation(); onMigrateSa3Runtime(); }} disabled={isBuilding}>
        migrate runtime
      </button>
    {/if}
    </div>{/if}
  </div>
</div>

<style>
  .service-row {
    padding: 10px 12px;
    margin: 2px 0;
    border: 1px solid transparent;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.1s;
  }
  .service-row:hover {
    background: var(--bg-hover);
  }
  .service-row.selected {
    background: var(--bg-panel);
    border-color: var(--border);
  }
  .row-top {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .info {
    display: flex;
    align-items: baseline;
    gap: 8px;
    flex: 1;
  }
  .name {
    font-weight: 600;
    font-size: 13px;
  }
  .meta {
    font-size: 11px;
    color: var(--text-secondary);
    font-family: var(--font-mono);
  }
  .error {
    margin: 4px 0 0 18px;
    font-size: 11px;
    color: var(--red);
    user-select: text;
    -webkit-user-select: text;
  }
  .build-done {
    margin: 4px 0 0 18px;
    font-size: 11px;
    color: var(--green);
  }
  .blocker {
    margin: 4px 0 0 18px;
    font-size: 11px;
    color: var(--yellow);
  }
  .build-progress {
    margin: 6px 0 0 18px;
  }
  .progress-bar {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 4px;
  }
  .progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.3s ease;
  }
  .build-label {
    font-size: 10px;
    color: var(--text-secondary);
    font-family: var(--font-mono);
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 8px;
    padding-left: 12px;
  }
  .controls button {
    flex: 0 0 auto;
    white-space: nowrap;
    font-size: 10px;
    padding: 3px 6px;
  }
  .runtime-actions { display: flex; gap: 4px; }
  .controls .models-btn,
  .controls .lora-btn {
    border-color: var(--accent);
    background: var(--accent);
    color: white;
  }
  .controls .models-btn:hover:not(:disabled),
  .controls .lora-btn:hover:not(:disabled) {
    filter: brightness(1.12);
  }
</style>
