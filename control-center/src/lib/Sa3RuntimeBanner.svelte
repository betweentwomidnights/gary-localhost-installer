<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import NativeBackendPicker from "./NativeBackendPicker.svelte";

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
  let saving = $state(false);
  let message: string | null = $state(null);
  let running = $derived(["running", "starting", "unhealthy"].includes(serviceStatus));
  let device = $derived(info?.devices.find((entry) => entry.deviceType !== "cpu" && entry.backend.toLowerCase().includes(info?.backend ?? "")));
  let wantsReinstall = $derived(!!info?.installed && !info.devOverride && info.preference !== "auto" && info.preference !== info.backend);

  async function load() {
    try { info = await invoke<RuntimeInfo>("get_native_runtime_info", { serviceId: "sa3" }); }
    catch (error) { message = String(error); }
  }
  $effect(() => { void nativeBackend; void envExists; void building; void load(); });

  async function choose(backend: string) {
    saving = true; message = null;
    try {
      await invoke("save_app_settings", { settings: { nativeBackends: { sa3: backend } } });
      await load();
      message = info?.installed ? "Reinstall the runtime to switch backends." : "Install the runtime to use this backend.";
    } catch (error) { message = String(error); }
    finally { saving = false; }
  }
  async function reinstall() {
    saving = true; message = null;
    try { await invoke("rebuild_env", { serviceId: "sa3" }); }
    catch (error) { message = String(error); }
    finally { saving = false; }
  }
</script>

<div class="runtime-banner">
  <div class="banner-row">
    <div class="copy">
      <div class="title">SA3 runtime</div>
      <div class="subtitle">
        {#if !info}checking…
        {:else if info.devOverride}local build on {info.backend}
        {:else if info.installed}{info.backend} · {info.version}{#if device} · {device.name} ({(device.memoryTotalBytes / 1024 ** 3).toFixed(1)} GB){/if}
        {:else}not installed yet. use install runtime on the left.
        {/if}
      </div>
    </div>
    {#if info}<NativeBackendPicker value={info.preference} backends={info.offeredBackends} disabled={saving || building} onChange={choose} />{/if}
  </div>
  {#if info?.fallbackReason}<div class="warning">{info.fallbackReason}</div>{/if}
  {#if wantsReinstall}
    <div class="note">Installed on {info?.backend}, set to {info?.preference}.
      {#if running}Stop SA3, then reinstall the runtime to switch.
      {:else}<button class="link" disabled={saving || building} onclick={reinstall}>reinstall now</button>{/if}
    </div>
  {/if}
  <div class="note">Auto selects the backend for your hardware. Choose a backend above to override it. Generation tiers are in <button class="link" onclick={onShowModels}>models</button>.</div>
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
