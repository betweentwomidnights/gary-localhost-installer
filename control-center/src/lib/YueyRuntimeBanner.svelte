<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import NativeBackendPicker from "./NativeBackendPicker.svelte";
  import { chooseYueyTier, loadYueyTiers, type YueyTiers } from "./yueyTiers";

  interface NativeDevice {
    name: string;
    backend: string;
    deviceType: string;
    memoryTotalBytes: number;
  }

  interface NativeRuntimeInfo {
    installed: boolean;
    devOverride: boolean;
    version: string | null;
    manifestVersion: string;
    backend: string | null;
    preference: string;
    fallbackReason: string | null;
    offeredBackends: string[];
    devices: NativeDevice[];
    recommendedEncoding: string | null;
  }

  let {
    serviceStatus,
    nativeBackend,
    envExists,
    building,
    startBlocker,
    onShowModels,
  }: {
    serviceStatus: "stopped" | "starting" | "running" | "unhealthy" | "failed";
    nativeBackend: string | null;
    envExists: boolean;
    building: boolean;
    startBlocker: string | null;
    onShowModels: () => void;
  } = $props();

  let info = $state<NativeRuntimeInfo | null>(null);
  let tiers = $state<YueyTiers | null>(null);
  let saving = $state(false);
  let message: string | null = $state(null);

  async function load() {
    try {
      info = await invoke<NativeRuntimeInfo>("get_native_runtime_info", { serviceId: "yuey" });
      tiers = await loadYueyTiers();
    } catch (e) {
      console.error("Failed to load the yuey runtime:", e);
    }
  }

  // Reload whenever an install finishes, the installed backend changes, or a
  // download lets yuey start.
  $effect(() => {
    void nativeBackend;
    void envExists;
    void building;
    void startBlocker;
    load();
  });

  let running = $derived(
    serviceStatus === "running" || serviceStatus === "starting" || serviceStatus === "unhealthy"
  );
  // The accelerator the runtime check found for the installed backend.
  let device = $derived(
    info?.devices.find(
      (d) => d.deviceType !== "cpu" && d.backend.toLowerCase().includes(info?.backend ?? "")
    ) ?? null
  );
  let wantsReinstall = $derived(
    !!info?.installed &&
      !info.devOverride &&
      info.preference !== "auto" &&
      info.preference !== info.backend
  );

  function gib(bytes: number): string {
    return (bytes / 1024 ** 3).toFixed(1);
  }

  async function choose(backend: string) {
    saving = true;
    message = null;
    try {
      await invoke("save_app_settings", { settings: { nativeBackends: { yuey: backend } } });
      await load();
      message = info?.installed
        ? "reinstall the runtime to switch backends."
        : "install the runtime to use this backend.";
    } catch (e: any) {
      message = "Failed: " + (typeof e === "string" ? e : e?.message || "unknown");
    } finally {
      saving = false;
    }
  }

  async function chooseTier(encoding: string) {
    saving = true;
    message = null;
    try {
      message = await chooseYueyTier(encoding);
      await load();
    } catch (e: any) {
      message = "Failed: " + (typeof e === "string" ? e : e?.message || "unknown");
    } finally {
      saving = false;
    }
  }

  async function downloadModels() {
    saving = true;
    message = null;
    try {
      const queued = await invoke<string[]>("download_yuey_default_models");
      message = queued.length
        ? `downloading ${queued.map((id) => id.replace("yuey::", "")).join(" and ")}. progress is in models.`
        : "already downloading. progress is in models.";
    } catch (e: any) {
      message = "Failed: " + (typeof e === "string" ? e : e?.message || "unknown");
    } finally {
      saving = false;
    }
  }

  async function reinstall() {
    message = null;
    try {
      await invoke("rebuild_env", { serviceId: "yuey" });
    } catch (e: any) {
      message = "Failed: " + (typeof e === "string" ? e : e?.message || "unknown");
    }
  }
</script>

<div class="yuey-banner">
  <div class="banner-row">
    <div class="copy">
      <div class="banner-title">yuey runtime</div>
      <div class="banner-subtitle">
        {#if !info}
          checking...
        {:else if info.devOverride}
          local build on {info.backend}
        {:else if info.installed}
          {info.backend} &middot; {info.version}{#if device} &middot; {device.name} ({gib(device.memoryTotalBytes)} GB){/if}
        {:else}
          not installed yet. use install runtime on the left.
        {/if}
      </div>
    </div>
    {#if tiers && tiers.installed.length > 1}
      <!-- Only downloaded tiers: the others are one download away in models. -->
      <label class="backend">
        <span>model</span>
        <select
          value={tiers.installed.includes(tiers.chosen) ? tiers.chosen : ""}
          disabled={saving}
          onchange={(e) => chooseTier((e.currentTarget as HTMLSelectElement).value)}
        >
          <option value="">auto{#if !tiers.installed.includes(tiers.chosen) && tiers.active} ({tiers.active}){/if}</option>
          {#each tiers.installed as tier}
            <option value={tier}>{tier}</option>
          {/each}
        </select>
      </label>
    {/if}
    {#if info}
      <NativeBackendPicker value={info.preference} backends={info.offeredBackends} disabled={saving || building} onChange={choose} />
    {/if}
  </div>

  {#if info?.fallbackReason}
    <div class="warning">{info.fallbackReason}</div>
  {/if}

  {#if info?.installed && startBlocker}
    <div class="note">
      yuey {startBlocker}.
      <button class="link" disabled={saving} onclick={downloadModels}>
        download the recommended models
      </button>
    </div>
  {/if}

  {#if wantsReinstall}
    <div class="note">
      installed on {info?.backend}, set to {info?.preference}.
      {#if running}
        stop yuey, then reinstall the runtime to switch.
      {:else}
        <button class="link" disabled={building} onclick={reinstall}>reinstall now</button>
      {/if}
    </div>
  {/if}

  <div class="note">
    auto picks CUDA on NVIDIA and Vulkan on AMD and Intel. on NVIDIA, CUDA is
    the faster of the two, and Vulkan has stalled on long renders with a DAW
    open.
    yuey still needs its models:
    <button class="link" onclick={onShowModels}>open models</button>
    {#if info?.recommendedEncoding}
      (this GPU suits {info.recommendedEncoding})
    {/if}
    {#if tiers && tiers.installed.length > 1}
      auto launches with the recommended tier when it is downloaded; pick
      another under model to trade quality for memory.
    {/if}
  </div>

  {#if message}
    <div class="msg">{message}</div>
  {/if}
</div>

<style>
  .yuey-banner {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: #1a2028;
  }

  .banner-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .copy {
    flex: 1;
    min-width: 0;
  }

  .banner-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .banner-subtitle {
    margin-top: 3px;
    font-size: 11px;
    color: var(--text-secondary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .backend {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-primary);
    white-space: nowrap;
  }

  .backend select {
    font-size: 11px;
  }

  .warning {
    margin-top: 8px;
    padding: 6px 8px;
    font-size: 10px;
    line-height: 1.4;
    color: var(--orange);
    border: 1px solid var(--orange);
    border-radius: 3px;
  }

  .note {
    margin-top: 8px;
    font-size: 10px;
    color: var(--text-muted);
    line-height: 1.4;
  }

  .link {
    padding: 0;
    border: none;
    background: none;
    color: var(--accent);
    font-size: 10px;
    cursor: pointer;
    text-decoration: underline;
  }

  .link:disabled {
    color: var(--text-muted);
    cursor: default;
  }

  .msg {
    margin-top: 6px;
    font-size: 10px;
    color: var(--text-secondary);
  }
</style>
