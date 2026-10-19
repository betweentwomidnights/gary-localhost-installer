<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import NativeBackendPicker from "./NativeBackendPicker.svelte";

  // A migrated service's own C++ runtime: what is installed on which GPU, the
  // backend it should use, and installing or updating it. Services that share
  // a runtime show this in their group's panel instead.
  interface RuntimeInfo {
    installed: boolean;
    devOverride: boolean;
    version: string | null;
    manifestVersion: string;
    backend: string | null;
    preference: string;
    fallbackReason: string | null;
    offeredBackends: string[];
    devices: { name: string; backend: string; deviceType: string; memoryTotalBytes: number }[];
  }
  let { serviceId, label, serviceStatus, nativeBackend, building, onShowModels }: {
    serviceId: string;
    label: string;
    serviceStatus: string;
    nativeBackend: string | null;
    building: boolean;
    onShowModels: () => void;
  } = $props();
  let info = $state<RuntimeInfo | null>(null);
  let busy = $state(false);
  let message = $state<string | null>(null);
  let error = $state<string | null>(null);
  const device = $derived(info?.devices.find((entry) => entry.deviceType !== "cpu" && entry.backend.toLowerCase().includes(info?.backend ?? "")));
  const action = $derived(!info || info.devOverride ? null
    : !info.installed ? "install runtime"
    : info.version !== info.manifestVersion ? "update runtime"
    : "reinstall runtime");
  const switching = $derived(!!info?.installed && !info.devOverride && info.preference !== "auto" && info.preference !== info.backend);
  const live = $derived(["running", "starting", "unhealthy"].includes(serviceStatus));

  async function load() {
    try { info = await invoke<RuntimeInfo>("get_native_runtime_info", { serviceId }); }
    catch (cause) { error = String(cause); }
  }
  $effect(() => { void nativeBackend; void building; void load(); });

  async function choose(backend: string) {
    busy = true; message = null; error = null;
    try {
      await invoke("save_app_settings", { settings: { nativeBackends: { [serviceId]: backend } } });
      await load();
      message = info?.installed && info.preference !== info.backend
        ? `Reinstall the runtime to switch ${label} to ${backend}.`
        : null;
    } catch (cause) { error = String(cause); }
    finally { busy = false; }
  }
  async function install() {
    busy = true; message = null; error = null;
    try { await invoke("install_shared_runtime", { serviceId }); }
    catch (cause) { error = String(cause); }
    finally { busy = false; }
  }
</script>

<div class="runtime-banner">
  <div class="banner-row">
    <div class="copy">
      <div class="title">{label} runtime</div>
      <div class="subtitle">
        {#if !info}checking…
        {:else if info.devOverride}local build on {info.backend}
        {:else if info.installed}{info.version} · {info.backend}{#if device} · {device.name} ({(device.memoryTotalBytes / 1024 ** 3).toFixed(1)} GB){/if}
        {:else}not installed yet{/if}
      </div>
    </div>
    {#if info}<NativeBackendPicker value={info.preference} backends={info.offeredBackends} disabled={busy || building} onChange={choose} />{/if}
  </div>
  {#if info?.fallbackReason}<div class="warning">{info.fallbackReason}</div>{/if}
  <div class="actions">
    {#if action}<button class="primary" disabled={busy || building} onclick={install}>{building ? "installing…" : action}</button>{/if}
    <span class="note">
      {#if info && info.installed && !info.devOverride && info.version !== info.manifestVersion}{info.manifestVersion} is available.
      {:else if switching}Installed on {info?.backend}, set to {info?.preference}: reinstall to switch.
      {:else if info?.installed}Up to date.{/if}
      {#if live && action} Stops {label} first.{/if}
      Model precisions are in <button class="link" onclick={onShowModels}>models</button>.
    </span>
  </div>
  {#if message}<div class="note" role="status">{message}</div>{/if}
  {#if error}<div class="error" role="alert">{error}</div>{/if}
</div>

<style>
  .runtime-banner { padding: 10px 16px; border-bottom: 1px solid var(--border); background: #1a2028; }
  .banner-row { display: flex; align-items: center; gap: 12px; }
  .copy { flex: 1; min-width: 0; }
  .title { font-size: 12px; font-weight: 600; color: var(--text-primary); }
  .subtitle { margin-top: 3px; font-size: 11px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .actions { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
  .actions button.primary { font-size: 11px; flex-shrink: 0; color: var(--accent); border-color: var(--accent); }
  .note { font-size: 10px; color: var(--text-muted); line-height: 1.4; }
  .warning { margin-top: 8px; padding: 6px 8px; font-size: 10px; line-height: 1.4; color: var(--orange); border: 1px solid var(--orange); border-radius: 3px; }
  .error { margin-top: 8px; font-size: 11px; color: var(--red); overflow-wrap: anywhere; }
  .link { padding: 0; border: none; background: none; color: var(--accent); font-size: 10px; cursor: pointer; text-decoration: underline; }
</style>
