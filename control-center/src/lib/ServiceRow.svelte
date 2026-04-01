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
  }

  let { service, selected, onSelect, hasModels = false, onShowModels = () => {} }: {
    service: ServiceInfo;
    selected: boolean;
    onSelect: () => void;
    hasModels?: boolean;
    onShowModels?: () => void;
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
      <span class="meta">:{service.port} {#if service.pid}&middot; PID {service.pid}{/if}</span>
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
    <div class="build-done">build complete</div>
  {/if}

  <div class="controls">
    {#if service.status === "stopped" || service.status === "failed"}
      <button onclick={(e) => { e.stopPropagation(); startService(); }} disabled={!service.env_exists || isBuilding}>start</button>
    {:else}
      <button onclick={(e) => { e.stopPropagation(); stopService(); }}>stop</button>
      <button onclick={(e) => { e.stopPropagation(); restartService(); }}>restart</button>
    {/if}
    <button onclick={(e) => { e.stopPropagation(); rebuildEnv(); }} disabled={isBuilding}>
      {#if isBuilding}
        building...
      {:else}
        {service.env_exists ? "rebuild env" : "build env"}
      {/if}
    </button>
    {#if hasModels}
      <button class="models-btn" onclick={(e) => { e.stopPropagation(); onShowModels(); }} disabled={!service.env_exists}>
        models
      </button>
    {/if}
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
    gap: 6px;
    margin-top: 8px;
    padding-left: 18px;
  }
  .controls button {
    font-size: 11px;
    padding: 3px 10px;
  }
  .controls .models-btn {
    border-color: var(--accent);
    color: var(--accent);
  }
  .controls .models-btn:hover:not(:disabled) {
    background: var(--accent);
    color: white;
  }
</style>
