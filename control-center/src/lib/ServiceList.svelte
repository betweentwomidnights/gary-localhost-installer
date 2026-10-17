<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import ServiceRow from "./ServiceRow.svelte";
  import { runtimeGroupLabel } from "./runtimeGroups";

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
    migration_complete: boolean;
    has_migration: boolean;
    runtime_group: string | null;
  }

  let { services, selectedServiceId, selectedRuntimeGroup, hfTokenConfigured, onSelect, onSelectGroup, onShowModels, onManageCareyLoras, onTrainCareyAce, onManageSa3Loras, onTrainSa3Lora, onMigrateRuntime }: {
    services: ServiceInfo[];
    selectedServiceId: string | null;
    selectedRuntimeGroup: string | null;
    onSelectGroup: (group: string) => void;
    hfTokenConfigured: boolean;
    onSelect: (id: string) => void;
    onShowModels: (id: string) => void;
    onManageCareyLoras: () => void;
    onTrainCareyAce: () => void;
    onManageSa3Loras: () => void;
    onTrainSa3Lora: () => void;
    onMigrateRuntime: (id: string) => void;
  } = $props();

  async function rebuildAll() {
    try {
      await invoke("rebuild_all_envs");
    } catch (e) {
      console.error("Rebuild all failed:", e);
    }
  }

  // Services that have downloadable models
  const servicesWithModels = new Set(["gary", "melodyflow", "stable-audio", "sa3", "carey", "foundation", "yuey"]);

  // Services sharing a native runtime are listed together under it, where
  // the first of them would have been.
  type Entry = { service: ServiceInfo } | { group: string; members: ServiceInfo[] };
  const entries = $derived.by(() => {
    const out: Entry[] = [];
    const seen = new Set<string>();
    for (const service of services) {
      const group = service.runtime_group;
      if (!group) { out.push({ service }); continue; }
      if (seen.has(group)) continue;
      seen.add(group);
      out.push({ group, members: services.filter((member) => member.runtime_group === group) });
    }
    return out;
  });
  const live = (service: ServiceInfo) => ["running", "starting", "unhealthy"].includes(service.status);
  function groupSummary(members: ServiceInfo[]) {
    const native = members.find((member) => member.runtime === "native");
    const parts = [`${members.filter(live).length}/${members.length} running`];
    if (native?.native_backend) parts.push(native.native_backend);
    if (members.some((member) => member.runtime === "native" && member.native_update_available)) parts.push("update available");
    if (members.some((member) => member.build_status?.building)) parts.push("busy");
    return parts.join(" · ");
  }
</script>

{#snippet row(service: ServiceInfo, grouped: boolean)}
  <ServiceRow
    {service}
    {grouped}
    selected={selectedServiceId === service.id}
    onSelect={() => onSelect(service.id)}
    hasModels={servicesWithModels.has(service.id) && (!["stable-audio", "sa3"].includes(service.id) || hfTokenConfigured)}
    onShowModels={() => onShowModels(service.id)}
    hasCareyLoras={service.id === "carey"}
    onManageCareyLoras={onManageCareyLoras}
    hasCareyAceTraining={service.id === "carey"}
    onTrainCareyAce={onTrainCareyAce}
    hasSa3Loras={service.id === "sa3"}
    onManageSa3Loras={onManageSa3Loras}
    hasSa3LoraTraining={service.id === "sa3" && hfTokenConfigured}
    onTrainSa3Lora={onTrainSa3Lora}
    hasMigration={service.has_migration && !service.migration_complete}
    onMigrateRuntime={() => onMigrateRuntime(service.id)}
  />
{/snippet}

<div class="service-list">
  <div class="list-header">
    <span class="label">services</span>
    <button onclick={rebuildAll}>rebuild all envs</button>
  </div>
  {#each entries as entry ("group" in entry ? `group:${entry.group}` : entry.service.id)}
    {#if "group" in entry}
      <div class="runtime-group">
        <button class="group-header" class:selected={selectedRuntimeGroup === entry.group} onclick={() => onSelectGroup(entry.group)}>
          <span class="group-name">{runtimeGroupLabel(entry.group)}</span>
          <span class="group-meta">{groupSummary(entry.members)}</span>
        </button>
        <div class="group-members">
          {#each entry.members as service (service.id)}{@render row(service, true)}{/each}
        </div>
      </div>
    {:else}
      {@render row(entry.service, false)}
    {/if}
  {/each}
  {#if services.length === 0}
    <div class="empty">loading services...</div>
  {/if}
</div>

<style>
  .service-list {
    padding: 8px;
  }
  .list-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    margin-bottom: 4px;
  }
  .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary);
    font-weight: 600;
  }
  .empty {
    padding: 20px;
    text-align: center;
    color: var(--text-muted);
  }
  .runtime-group { margin: 6px 0; }
  .group-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
    width: 100%;
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: transparent;
    color: var(--text-primary);
    text-align: left;
    cursor: pointer;
  }
  .group-header:hover { background: var(--bg-hover); }
  .group-header.selected { background: var(--bg-panel); border-color: var(--accent); }
  .group-name { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
  .group-meta { font-size: 10px; color: var(--text-secondary); font-family: var(--font-mono); white-space: nowrap; }
  .group-members { margin-left: 10px; padding-left: 4px; border-left: 1px solid var(--border); }
</style>
