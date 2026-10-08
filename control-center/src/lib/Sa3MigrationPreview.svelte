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
    status: string;
  }

  interface MigrationItem {
    label: string;
    path: string;
    bytes: number;
    kind: "code" | "environment" | "weights" | null;
  }
  interface MigrationPreview {
    activeRoot: string;
    hfHubCache: string;
    cleanupCandidates: MigrationItem[];
    estimatedCleanupBytes: number;
    preservedPaths: MigrationItem[];
    warnings: string[];
    nativeSelection: { encoding: string; verifiedRelease: string; cleanupComplete: boolean; cleanupErrors: string[] } | null;
    cleanupToken: string;
  }
  type NativeSelection = NonNullable<MigrationPreview["nativeSelection"]>;

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
  let activating = $state(false);
  let serviceRunning = $state(false);
  let activeEncoding = $state("F16");
  let encodingTouched = $state(false);
  let selection: NativeSelection | null = $state(null);
  let activationMessage: string | null = $state(null);
  let activationProgress = $state("");
  let cleaning = $state(false);
  let validatedClient = $state(false);

  function updateBuild(services: ServiceStatus[]) {
    const service = services.find((service) => service.id === "sa3");
    serviceRunning = ["running", "starting", "unhealthy"].includes(service?.status ?? "");
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
    const migrationListener = listen<string>("sa3-native-migration-progress", (event) => { if (!disposed) activationProgress = event.payload; });
    void Promise.all([refreshRuntime(), invoke<ServiceStatus[]>("get_services").then((services) => {
      if (!disposed) updateBuild(services);
    })]).catch((cause) => { if (!disposed) preparationError = String(cause); });
    void invoke<NativeSelection | null>("get_sa3_native_runtime_selection").then((saved) => {
      if (disposed) return;
      selection = saved;
      if (saved && !encodingTouched) activeEncoding = saved.encoding;
    }).catch((cause) => { if (!disposed) error = String(cause); });
    return () => { disposed = true; void unlisten.then((stop) => stop()); void migrationListener.then((stop) => stop()); };
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
      selection = preview.nativeSelection;
    } catch (cause) {
      error = String(cause);
    } finally {
      busy = false;
    }
  }

  const codeCandidates = $derived.by((): MigrationItem[] => preview?.cleanupCandidates.filter((entry) => entry.kind === "code") ?? []);
  const otherCandidates = $derived.by((): MigrationItem[] => preview?.cleanupCandidates.filter((entry) => entry.kind !== "code") ?? []);

  async function cleanup() {
    if (!preview || !selection || !validatedClient) return;
    const paths = otherCandidates.map((entry) => `${entry.label}: ${entry.path}`).join("\n");
    const codeSummary = codeCandidates.length ? `\n\n${codeCandidates.length} unchanged bundled Python files listed in the review under ${preview.activeRoot}/services/sa3.` : "";
    if (!window.confirm(`Retire these SA3 Python files from ${preview.activeRoot}?\n\n${paths || (codeCandidates.length ? "Bundled Python source only." : "No legacy files remain.")}${codeSummary}\n\nEstimated size: ${formatBytes(preview.estimatedCleanupBytes)}. This removal cannot be undone. Native verification runs again before cleanup.`)) return;
    cleaning = true; error = null; activationMessage = null;
    activationProgress = "Verifying the native runtime before cleanup...";
    try {
      const result = await invoke<{selection:NativeSelection; errors:string[]; estimatedRemovedBytes:number}>("cleanup_sa3_legacy_installation", { reviewToken:preview.cleanupToken });
      selection = result.selection;
      if (result.errors.length) error = result.errors.join("\n");
      activationMessage = result.selection.cleanupComplete ? `Legacy SA3 cleanup completed (${formatBytes(result.estimatedRemovedBytes)} estimated). C++ remains selected.` : "Some legacy files remain. C++ stays selected; rescan and retry cleanup.";
      preview = await invoke<MigrationPreview>("get_sa3_migration_preview");
    } catch (cause) { error = String(cause); }
    finally { cleaning = false; }
  }

  async function activate() {
    activating = true;
    error = null;
    activationMessage = null;
    activationProgress = "Checking prepared runtime and models...";
    try {
      selection = await invoke<NativeSelection>("activate_sa3_native_runtime", { encoding: activeEncoding });
      activationMessage = "C++ selected for this storage profile. Python files are preserved until cleanup.";
      await scan();
    } catch (cause) { error = String(cause); }
    finally { activating = false; }
  }
</script>

<section aria-labelledby="sa3-migration-title">
  <div class="heading">
    <div>
      <h3 id="sa3-migration-title">SA3 native migration preflight</h3>
      <p>Review the storage used by the current SA3 Python installation.</p>
    </div>
    <button type="button" onclick={scan} disabled={busy || cleaning || activating || pendingRestart}>
      {busy ? "scanning..." : preview ? "rescan SA3" : "preview SA3 cleanup"}
    </button>
  </div>
  <p class="note">Prepare the runtime, models and adapters, then verify and select C++ for this storage profile. Test generation and your LoRAs before retiring Python.</p>
  <div class="preparation">
    <div class="heading">
      <div>
        <h3>Prepare the C++ runtime</h3>
        <p>Download and verify the shared SA3 runtime using your detected GPU. Your Python service and models stay available.</p>
      </div>
      <button type="button" onclick={prepareRuntime} disabled={preparing || serviceBuilding || cleaning || pendingRestart}>
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
  <div class="preparation">
    <div class="heading">
      <div><h3>Use the C++ runtime</h3><p>Checks trainer capabilities, verifies model and adapter files, and tests native generation before switching SA3.</p></div>
      <button type="button" onclick={activate} disabled={activating || preparing || serviceBuilding || busy || pendingRestart || serviceRunning || !runtime?.installed}>{activating ? "validating and switching..." : "verify and select C++"}</button>
    </div>
    <label>Inference model precision
      <select bind:value={activeEncoding} disabled={activating || pendingRestart} onchange={() => encodingTouched = true}>
        <option value="F16">F16</option><option value="Q8_0">Q8_0</option><option value="Q5_K_M">Q5_K_M</option><option value="Q4_K_M">Q4_K_M</option>
      </select>
    </label>
    <p class="note">Choose a prepared model. Training base precision is selected separately. Stop SA3 before switching.</p>
    {#if activating}<p class="note" role="status">{activationProgress}</p>{/if}
    {#if selection}<p class="note" role="status">C++ selected: {selection.encoding}, release {selection.verifiedRelease}. {selection.cleanupComplete ? "Legacy cleanup completed." : "Legacy cleanup pending."}</p>{/if}
    {#if selection}{#each selection.cleanupErrors ?? [] as issue}<p class="note">{issue}</p>{/each}{/if}
    {#if activationMessage}<p class="note" role="status">{activationMessage}</p>{/if}
  </div>
  {#if pendingRestart}
    <p class="note">Restart to use your chosen storage folder before scanning.</p>
  {:else if preview}
    <div class="summary">{formatBytes(preview.estimatedCleanupBytes)} in potential cleanup</div>
    <p class="note">Estimates include cached weight copies and may differ from disk space recovered. Old storage profiles use the existing cleanup section below.</p>
    <div class="label">active storage</div>
    <button class="path" type="button" onclick={() => preview && onReveal(preview.activeRoot)}>{preview.activeRoot}</button>
    <div class="label">effective Hugging Face cache</div>
    <button class="path" type="button" onclick={() => preview && onReveal(preview.hfHubCache)}>{preview.hfHubCache}</button>
    <div class="cleanup-list">
    {#each otherCandidates as entry (entry.path)}
      <div class="entry">
        <div class="entry-heading"><span>{entry.label}</span><span>{formatBytes(entry.bytes)}</span></div>
        <button class="path" type="button" onclick={() => onReveal(entry.path)}>{entry.path}</button>
      </div>
    {/each}
    {#if codeCandidates.length}
      <details>
        <summary>{codeCandidates.length} unchanged bundled Python source files · {formatBytes(codeCandidates.reduce((sum, entry) => sum + entry.bytes, 0))}</summary>
        {#each codeCandidates as entry (entry.path)}
          <div class="entry"><button class="path" type="button" onclick={() => onReveal(entry.path)}>{entry.path}</button></div>
        {/each}
      </details>
    {/if}
    {#if !preview.cleanupCandidates.length}<p class="note">No reviewed legacy SA3 environments, code or model caches found in this storage profile.</p>{/if}
    </div>
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
    {#if selection && (!selection.cleanupComplete || preview.cleanupCandidates.length > 0)}
      <div class="preparation">
        <h3>Retire the SA3 Python installation</h3>
        <p>Removes the reviewed SA3 Python environments, unchanged bundled Python code and PyTorch weight repositories. Original LoRAs, checkpoints, prompts, native models, shared runtimes and other services are preserved. Edited or unrecognized service files, settings and developer source checkouts are kept.</p>
        <label class="confirmation"><input type="checkbox" bind:checked={validatedClient} disabled={cleaning} /> I have tested native generation and my LoRAs in Gary, including training where needed.</label>
        <button type="button" onclick={cleanup} disabled={cleaning || activating || preparing || serviceBuilding || serviceRunning || busy || pendingRestart || !validatedClient || preview.warnings.length > 0}>{cleaning ? "verifying and cleaning up..." : "clean up reviewed Python files"}</button>
        {#if cleaning}<p class="note" role="status">{activationProgress}</p>{/if}
      </div>
    {/if}
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
  label { font-size: 12px; }
  select { margin: 6px 0 6px 8px; background: var(--bg-secondary, #222); border: 1px solid var(--border, #444); color: inherit; padding: 6px; border-radius: 6px; }
  .entry { margin-top: 12px; font-size: 12px; }
  .cleanup-list { max-height: 320px; overflow: auto; }
  details { margin-top: 16px; font-size: 12px; }
  summary { cursor: pointer; }
  .preserved-path { margin-top: 3px; color: var(--text-secondary, #aaa); overflow-wrap: anywhere; }
  .error { color: var(--error, #f99); }
  .confirmation { display: flex; align-items: start; gap: 8px; margin: 12px 0; line-height: 1.5; }
</style>
