<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";
  import Sa3NativeModels from "./Sa3NativeModels.svelte";
  import Sa3NativeLoras from "./Sa3NativeLoras.svelte";

  interface NativeRuntimeInfo {
    installed: boolean;
    version: string | null;
    backend: string | null;
    fallbackReason: string | null;
  }
  interface ServiceStatus {
    id: string;
    build_status: { building: boolean; step_label: string } | null;
  }

  interface MigrationItem {
    label: string;
    path: string;
    bytes: number;
  }
  interface MigrationPreview {
    activeRoot: string;
    hfHubCache: string;
    cleanupCandidates: MigrationItem[];
    estimatedCleanupBytes: number;
    preservedPaths: MigrationItem[];
    warnings: string[];
  }

  let { pendingRestart, onReveal }: {
    pendingRestart: boolean;
    onReveal: (path: string) => void;
  } = $props();
  let preview: MigrationPreview | null = $state(null);
  let busy = $state(false);
  let error: string | null = $state(null);
  let runtime: NativeRuntimeInfo | null = $state(null);
  let preparing = $state(false);
  let serviceBuilding = $state(false);
  let preparationStep = $state("");
  let preparationError: string | null = $state(null);

  function updateBuild(services: ServiceStatus[]) {
    const status = services.find((service) => service.id === "sa3")?.build_status;
    serviceBuilding = status?.building ?? false;
    preparationStep = status?.step_label ?? "";
  }

  async function refreshRuntime() {
    runtime = await invoke<NativeRuntimeInfo>("get_native_runtime_info", { serviceId: "sa3" });
  }

  onMount(() => {
    let disposed = false;
    const unlisten = listen<ServiceStatus[]>("services-updated", (event) => {
      if (disposed) return;
      const wasBuilding = serviceBuilding;
      updateBuild(event.payload);
      if (wasBuilding && !serviceBuilding) {
        void refreshRuntime().catch((cause) => { preparationError = String(cause); });
      }
    });
    void Promise.all([refreshRuntime(), invoke<ServiceStatus[]>("get_services").then((services) => {
      if (!disposed) updateBuild(services);
    })]).catch((cause) => { if (!disposed) preparationError = String(cause); });
    return () => { disposed = true; void unlisten.then((stop) => stop()); };
  });

  async function prepareRuntime() {
    preparing = true;
    preparationError = null;
    preparationStep = "Preparing SA3 runtime...";
    try {
      runtime = await invoke<NativeRuntimeInfo>("prepare_sa3_native_runtime");
    } catch (cause) {
      preparationError = String(cause);
    } finally {
      preparing = false;
    }
  }

  function formatBytes(bytes: number) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
    return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  }

  async function scan() {
    busy = true;
    error = null;
    preview = null;
    try {
      preview = await invoke<MigrationPreview>("get_sa3_migration_preview");
    } catch (cause) {
      error = String(cause);
    } finally {
      busy = false;
    }
  }
</script>

<section aria-labelledby="sa3-migration-title">
  <div class="heading">
    <div>
      <h3 id="sa3-migration-title">SA3 native migration preflight</h3>
      <p>Review the storage used by the current SA3 Python installation.</p>
    </div>
    <button type="button" onclick={scan} disabled={busy || pendingRestart}>
      {busy ? "scanning..." : preview ? "rescan SA3" : "preview SA3 cleanup"}
    </button>
  </div>
  <p class="note">Migration will be available after native inference, LoRA conversion and training are validated. This preview only reads your storage.</p>
  <div class="preparation">
    <div class="heading">
      <div>
        <h3>Prepare the C++ runtime</h3>
        <p>Download and verify the shared SA3 runtime using your detected GPU. Your Python service and models stay available.</p>
      </div>
      <button type="button" onclick={prepareRuntime} disabled={preparing || serviceBuilding || pendingRestart}>
        {preparing || serviceBuilding ? "preparing..." : runtime?.installed ? "verify / reinstall runtime" : "prepare SA3 runtime"}
      </button>
    </div>
    {#if preparing || serviceBuilding}
      <p class="note" role="status">{preparationStep}</p>
    {:else if runtime?.installed}
      <p class="note" role="status">Runtime prepared: {runtime.version ?? "local build"}, {runtime.backend ?? "auto"}. Model setup and service migration are the next steps.</p>
      {#if runtime.fallbackReason}<p class="note">{runtime.fallbackReason}</p>{/if}
    {/if}
    {#if preparationError}<p class="error" role="alert">{preparationError}</p>{/if}
  </div>
  <Sa3NativeModels {pendingRestart} />
  <Sa3NativeLoras {pendingRestart} />
  {#if pendingRestart}
    <p class="note">Restart to use your chosen storage folder before scanning.</p>
  {:else if preview}
    <div class="summary">{formatBytes(preview.estimatedCleanupBytes)} in potential cleanup</div>
    <p class="note">Estimates include cached weight copies and may differ from disk space recovered. Old storage profiles use the existing cleanup section below.</p>
    <div class="label">active storage</div>
    <button class="path" type="button" onclick={() => preview && onReveal(preview.activeRoot)}>{preview.activeRoot}</button>
    <div class="label">effective Hugging Face cache</div>
    <button class="path" type="button" onclick={() => preview && onReveal(preview.hfHubCache)}>{preview.hfHubCache}</button>
    {#each preview.cleanupCandidates as entry (entry.path)}
      <div class="entry">
        <div class="entry-heading"><span>{entry.label}</span><span>{formatBytes(entry.bytes)}</span></div>
        <button class="path" type="button" onclick={() => onReveal(entry.path)}>{entry.path}</button>
      </div>
    {:else}
      <p class="note">No legacy SA3 environments or model caches found in this storage profile.</p>
    {/each}
    <details>
      <summary>Preserved through migration</summary>
      {#each preview.preservedPaths as entry (entry.path)}
        <div class="entry">
          <div>{entry.label}</div>
          <div class="preserved-path">{entry.path}</div>
        </div>
      {/each}
      <p class="note">Settings, tokens, external datasets and other services stay in place. The UV cache is shared by Python services; clearing it remains a separate storage action.</p>
    </details>
    {#each preview.warnings as warning}
      <p class="error">{warning}</p>
    {/each}
  {/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
</section>

<style>
  section { margin: 20px 0; padding-top: 18px; border-top: 1px solid var(--border, #333); }
  .heading, .entry-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  h3 { margin: 0; font-size: 14px; font-weight: 600; }
  p { margin: 5px 0 10px; font-size: 12px; color: var(--text-secondary, #aaa); line-height: 1.5; }
  button { background: transparent; border: 1px solid var(--border, #444); color: inherit; border-radius: 6px; padding: 7px 10px; font: inherit; font-size: 12px; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
  .path { display: block; text-align: left; border: none; padding: 3px 0; color: var(--text-secondary, #aaa); overflow-wrap: anywhere; width: 100%; }
  .path:hover { text-decoration: underline; }
  .label { font-size: 11px; color: var(--text-secondary, #aaa); margin-top: 10px; }
  .summary { margin-top: 12px; font-size: 14px; }
  .preparation { margin: 16px 0; }
  .entry { margin-top: 12px; font-size: 12px; }
  details { margin-top: 16px; font-size: 12px; }
  summary { cursor: pointer; }
  .preserved-path { margin-top: 3px; color: var(--text-secondary, #aaa); overflow-wrap: anywhere; }
  .error { color: var(--error, #f99); }
</style>
