<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";

  interface RuntimeInfo {
    installed: boolean;
    devOverride: boolean;
    version: string | null;
    backend: string | null;
    preference: string;
    fallbackReason: string | null;
    offeredBackends: string[];
    devices: { name: string; backend: string; deviceType: string; memoryTotalBytes: number }[];
  }
  let { serviceStatus, nativeBackend, envExists, building, onShowModels }: {
    serviceStatus: string;
    nativeBackend: string | null;
    envExists: boolean;
    building: boolean;
    onShowModels: () => void;
  } = $props();
  let info = $state<RuntimeInfo | null>(null);
  let message: string | null = $state(null);
  let device = $derived(info?.devices.find((entry) => entry.deviceType !== "cpu" && entry.backend.toLowerCase().includes(info?.backend ?? "")));

  async function load() {
    try { info = await invoke<RuntimeInfo>("get_native_runtime_info", { serviceId: "sa3" }); }
    catch (error) { message = String(error); }
  }
  $effect(() => { void nativeBackend; void envExists; void building; void load(); });
</script>

<div class="runtime-banner">
  <div class="banner-row">
    <div class="copy">
      <div class="title">SA3 runtime</div>
      <div class="subtitle">
        {#if !info}checking…
        {:else if info.devOverride}local build on {info.backend}
        {:else if info.installed}{info.backend} · {info.version}{#if device} · {device.name} ({(device.memoryTotalBytes / 1024 ** 3).toFixed(1)} GB){/if}
        {:else}not installed yet. select stable audio on the left to install it.
        {/if}
      </div>
    </div>
  </div>
  {#if info?.fallbackReason}<div class="warning">{info.fallbackReason}</div>{/if}
  <div class="note">SA3 shares this runtime with jerry and foundation-1: pick its backend and install or update it under stable audio on the left. Generation tiers are in <button class="link" onclick={onShowModels}>models</button>.</div>
  {#if message}<div class="note" role="status">{message}</div>{/if}
</div>

<style>
  .runtime-banner { padding: 10px 16px; border-bottom: 1px solid var(--border); background: #1a2028; }
  .banner-row { display: flex; align-items: center; gap: 12px; }
  .copy { flex: 1; min-width: 0; }
  .title { font-size: 12px; font-weight: 600; color: var(--text-primary); }
  .subtitle { margin-top: 3px; font-size: 11px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .note { margin-top: 8px; font-size: 10px; color: var(--text-muted); line-height: 1.4; }
  .warning { margin-top: 8px; padding: 6px 8px; font-size: 10px; line-height: 1.4; color: var(--orange); border: 1px solid var(--orange); border-radius: 3px; }
  .link { padding: 0; border: none; background: none; color: var(--accent); font-size: 10px; cursor: pointer; text-decoration: underline; }
  .link:disabled { color: var(--text-muted); cursor: default; }
</style>
