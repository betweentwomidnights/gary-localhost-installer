<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import ServiceRow from "./ServiceRow.svelte";

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

  let { services, selectedServiceId, hfTokenConfigured, onSelect, onShowModels, onManageCareyLoras, onTrainCareyAce, onManageSa3Loras, onTrainSa3Lora }: {
    services: ServiceInfo[];
    selectedServiceId: string | null;
    hfTokenConfigured: boolean;
    onSelect: (id: string) => void;
    onShowModels: (id: string) => void;
    onManageCareyLoras: () => void;
    onTrainCareyAce: () => void;
    onManageSa3Loras: () => void;
    onTrainSa3Lora: () => void;
  } = $props();

  async function rebuildAll() {
    try {
      await invoke("rebuild_all_envs");
    } catch (e) {
      console.error("Rebuild all failed:", e);
    }
  }

  // Services that have downloadable models
  const servicesWithModels = new Set(["gary", "stable-audio", "sa3", "carey", "foundation"]);
</script>

<div class="service-list">
  <div class="list-header">
    <span class="label">services</span>
    <button onclick={rebuildAll}>rebuild all envs</button>
  </div>
  {#each services as service (service.id)}
    <ServiceRow
      {service}
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
    />
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
</style>
