<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount, tick } from "svelte";
  import { recommendQuantizedSa3Training, type Sa3TrainingHardware } from "./sa3NativeTraining";

  interface Item { label: string; path: string; bytes: number; kind: string | null; }
  interface Selection { encoding: string; verifiedRelease: string; cleanupComplete: boolean; cleanupErrors: string[]; }
  interface Preview {
    activeRoot: string;
    hfHubCache: string;
    cleanupCandidates: Item[];
    legacyRuntimePresent: boolean;
    trainingBasePresent: boolean;
    estimatedCleanupBytes: number;
    preservedPaths: Item[];
    warnings: string[];
    nativeSelection: Selection | null;
    cleanupToken: string;
  }
  interface Service { id: string; status: string; build_status: { building: boolean; step_label: string } | null; }
  interface Download { model_id: string; progress: number; status: string; message: string; error: string | null; }
  interface Component { id: string; label: string; files: { bytes: number }[]; }
  type Stage = "setup" | "cleanup" | "complete";

  let { open, pendingRestart, onClose, onReveal }: {
    open: boolean;
    pendingRestart: boolean;
    onClose: () => void;
    onReveal: (path: string) => void;
  } = $props();

  let preview: Preview | null = $state(null);
  let catalog: Component[] = $state([]);
  let stage: Stage = $state("setup");
  let loading = $state(false);
  let working = $state(false);
  let error: string | null = $state(null);
  let progress = $state("");
  let task = $state(-1);
  let downloads: Download[] = $state([]);
  let encoding = $state("F16");
  let trainingBase = $state("F16");
  let includeTraining = $state(true);
  let includeDecoder = $state(false);
  let elapsedSeconds = $state(0);
  let services: Service[] = $state([]);
  let dialog: HTMLDivElement | undefined = $state();
  const tasks = ["Prepare the C++ runtime", "Prepare model files", "Prepare your LoRAs", "Verify and switch to C++"];
  const stepNumber = $derived(stage === "setup" ? working && task === 3 ? 2 : 1 : 3);
  const blocked = $derived(loading || working || pendingRestart || !!services.find((entry) => entry.id === "sa3")?.build_status?.building);
  const modelIds = $derived([
    "sa3-native::text", "sa3-native::medium-decoder", `sa3-native::medium-${encoding}`,
    ...(includeTraining ? [`sa3-native::medium-base-${trainingBase}`] : []),
    ...(includeDecoder ? ["sa3-native::decoder-correction"] : []),
  ]);
  const modelBytes = $derived(catalog.filter((entry) => modelIds.includes(entry.id)).reduce((sum, entry) => sum + entry.files.reduce((bytes, file) => bytes + file.bytes, 0), 0));
  const hasCleanup = $derived.by(() => !!preview && (preview.cleanupCandidates.length > 0 || preview.legacyRuntimePresent || (preview.nativeSelection?.cleanupErrors.length ?? 0) > 0));
  const selectedDownloads = $derived(downloads.filter((entry) => modelIds.includes(entry.model_id)));
  const activeDownload = $derived(selectedDownloads.find((entry) => entry.status === "downloading" && /^(Downloading|Verifying)/.test(entry.message)) ?? selectedDownloads.find((entry) => entry.status === "downloading"));
  const preparedBytes = $derived(catalog.filter((entry) => modelIds.includes(entry.id)).reduce((sum, entry) => {
    const download = selectedDownloads.find((download) => download.model_id === entry.id);
    return sum + entry.files.reduce((bytes, file) => bytes + file.bytes, 0) * (download?.status === "downloaded" ? 1 : Math.min(1, Math.max(0, download?.progress ?? 0)));
  }, 0));

  function formatBytes(bytes: number) { return `${(bytes / 1024 ** 3).toFixed(1)} GB`; }

  async function scan() {
    preview = await invoke<Preview>("get_sa3_migration_preview");
  }

  async function initialize() {
    loading = true;
    preview = null;
    stage = "setup";
    error = null;
    downloads = [];
    task = -1;
    try {
      const [storage, hardware, components, currentServices, decoder] = await Promise.all([
        invoke<Preview>("get_sa3_migration_preview"),
        invoke<Sa3TrainingHardware>("get_native_runtime_info", { serviceId: "sa3" }),
        invoke<Component[]>("get_sa3_native_model_catalog"),
        invoke<Service[]>("get_services"),
        invoke<{ enabled: boolean }>("get_sa3_native_decoder_state"),
      ]);
      preview = storage;
      catalog = components;
      services = currentServices;
      encoding = storage.nativeSelection?.encoding ?? "F16";
      trainingBase = recommendQuantizedSa3Training(hardware) ? "Q4_K_M" : "F16";
      includeTraining = storage.trainingBasePresent;
      includeDecoder = decoder.enabled;
      stage = storage.nativeSelection?.cleanupComplete ? "complete" : "setup";
    } catch (cause) { error = String(cause); }
    finally { loading = false; }
  }

  $effect(() => {
    if (!open) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    void tick().then(() => dialog?.focus());
    void initialize();
    return () => { previousFocus?.focus(); };
  });

  $effect(() => {
    if (!working) return;
    const started = Date.now();
    elapsedSeconds = 0;
    const timer = setInterval(() => elapsedSeconds = Math.floor((Date.now() - started) / 1000), 1000);
    return () => clearInterval(timer);
  });
  $effect(() => {
    if (!working || task !== 1) return;
    let disposed = false;
    let polling = false;
    const timer = setInterval(async () => {
      if (polling) return;
      polling = true;
      try { const state = await invoke<Download[]>("get_download_progress"); if (!disposed) downloads = state; }
      catch { /* Live events continue to report transfer errors. */ }
      finally { polling = false; }
    }, 1000);
    return () => { disposed = true; clearInterval(timer); };
  });

  onMount(() => {
    let disposed = false;
    const listeners = [
      listen<Service[]>("services-updated", (event) => {
        if (disposed) return;
        services = event.payload;
        const build = services.find((entry) => entry.id === "sa3")?.build_status;
        if (working && task === 0 && build?.building) progress = build.step_label;
      }),
      listen<Download[]>("download-progress", (event) => { if (!disposed) downloads = event.payload; }),
      listen<string>("sa3-native-migration-progress", (event) => { if (!disposed && working) progress = event.payload; }),
    ];
    return () => { disposed = true; for (const listener of listeners) void listener.then((stop) => stop()); };
  });

  async function migrate() {
    if (blocked || !preview) return;
    working = true;
    error = null;
    downloads = [];
    try {
      task = 0;
      progress = "Stopping SA3 before migration...";
      const current = await invoke<Service[]>("get_services");
      if (current.some((entry) => entry.id === "sa3" && ["running", "starting", "unhealthy"].includes(entry.status))) {
        await invoke("stop_service", { serviceId: "sa3" });
      }
      progress = "Installing and checking the runtime for your hardware...";
      await invoke("prepare_sa3_native_runtime");
      task = 1;
      progress = "Downloading and verifying model files. Existing components are reused.";
      await invoke("prepare_sa3_native_models", { encoding, trainingBase: includeTraining ? trainingBase : null, includeDecoder, waitForCompletion: true });
      if (includeDecoder) {
        progress = "Preparing your decoder squeak fix...";
        await invoke("prepare_sa3_native_decoder");
      }
      task = 2;
      progress = "Preparing native copies of your registered LoRAs...";
      const loras = await invoke<{ entries: { name: string; nativePath: string | null; error: string | null }[] }>("prepare_sa3_native_loras");
      const failed = loras.entries.filter((entry) => !entry.nativePath || entry.error);
      if (failed.length) throw new Error(failed.map((entry) => `${entry.name}: ${entry.error ?? "Native copy is missing"}`).join("\n"));
      task = 3;
      progress = "Verifying the server, training tools, model files and native generation...";
      await invoke<Selection>("activate_sa3_native_runtime", { encoding });
      await scan();
      stage = hasCleanup ? "cleanup" : "complete";
    } catch (cause) { error = String(cause); }
    finally { working = false; }
  }

  async function cleanup() {
    if (blocked || !preview || preview.warnings.length) return;
    working = true;
    error = null;
    progress = "Verifying C++ again before removing reviewed Python files...";
    try {
      const current = await invoke<Service[]>("get_services");
      if (current.some((entry) => entry.id === "sa3" && ["running", "starting", "unhealthy"].includes(entry.status))) {
        await invoke("stop_service", { serviceId: "sa3" });
      }
      // The current review token binds this action to the paths shown below.
      const result = await invoke<{ selection: Selection; errors: string[] }>("cleanup_sa3_legacy_installation", { reviewToken: preview.cleanupToken });
      await scan();
      if (result.selection.cleanupComplete) stage = "complete";
      if (result.errors.length) error = result.errors.join("\n");
    } catch (cause) { error = String(cause); await scan().catch(() => {}); }
    finally { working = false; }
  }

  function close() { if (!working && !loading) onClose(); }
  function handleKeys(event: KeyboardEvent) {
    if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); close(); }
    if (event.key !== "Tab" || !dialog) return;
    const controls = Array.from(dialog.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), summary, [tabindex="0"]')).filter((element) => element.getClientRects().length);
    const first = controls[0], last = controls.at(-1);
    if (!first) { event.preventDefault(); return; }
    if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog)) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && (document.activeElement === last || document.activeElement === dialog)) { event.preventDefault(); first.focus(); }
  }
</script>

{#if open}
  <div class="overlay">
    <button class="backdrop" type="button" aria-label="Close runtime migration" onclick={close} disabled={working || loading}></button>
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="sa3-migration-title" aria-busy={working || loading} tabindex="-1" bind:this={dialog} onkeydown={handleKeys}>
      <header>
        <div><div class="eyebrow">SA3 runtime</div><h2 id="sa3-migration-title">{stage === "complete" ? "C++ is ready" : "Migrate SA3 to C++"}</h2></div>
        <button type="button" class="close" aria-label="Close runtime migration" onclick={close} disabled={working || loading}>×</button>
      </header>
      <ol class="steps" aria-label="Migration steps">
        {#each ["Prepare", "Automatic checks", "Clean up"] as label, index}
          <li class:current={stepNumber === index + 1} class:done={stepNumber > index + 1 || stage === "complete"} aria-current={stepNumber === index + 1 ? "step" : undefined}><span>{stepNumber > index + 1 || stage === "complete" ? "✓" : index + 1}</span>{label}</li>
        {/each}
      </ol>
      <div class="content">
        {#if loading}<p role="status">Checking your SA3 installation and storage…</p>
        {:else if preview}
          {#if stage === "setup"}
            {#if !working}
            <h3>One action to prepare and switch</h3>
            <p>Gary will prepare the runtime, models and your LoRAs, then automatically check generation{includeTraining ? ", training and checkpoint saving" : ""} before offering cleanup.</p>
            <p>{preview.legacyRuntimePresent ? "Your Python installation stays available until the checks pass and you choose to clean it up." : "Your LoRAs, datasets, prompts and other services stay in place."} SA3 will stop while migration runs.</p>
            <div class="summary">{encoding} inference{includeTraining ? ` · ${trainingBase} training` : ""}{modelBytes ? ` · ${formatBytes(modelBytes)} of model files` : ""}<small>Existing model components and shared runtimes are reused.</small></div>
            <details>
              <summary>Model options</summary>
              <div class="options">
                <label>Inference precision<select bind:value={encoding} disabled={blocked}><option>F16</option><option>Q8_0</option><option>Q5_K_M</option><option>Q4_K_M</option></select></label>
              </div>
              <p>{includeTraining ? `Your existing training setup was detected. Gary includes its ${trainingBase} C++ replacement automatically.` : "No existing training base was found. You can add training models later from Models."}</p>
            </details>
            {:else}
              <h3>{task === 3 ? "Running automatic checks" : "Preparing SA3 C++"}</h3>
              <p class="progress" role="status">{progress}</p>
              {#if task === 1}
                <div class="download-summary"><strong>{activeDownload ? catalog.find((entry) => entry.id === activeDownload.model_id)?.label ?? "Preparing model files" : "Checking model files"}</strong><span>{modelBytes ? `${Math.round(preparedBytes / modelBytes * 100)}%` : ""}</span></div>
                <progress max="1" value={modelBytes ? preparedBytes / modelBytes : undefined} aria-label="Overall model preparation"></progress>
                <p>{formatBytes(preparedBytes)} of {formatBytes(modelBytes)} prepared</p>
                <p class="file-progress" role="status">{activeDownload?.message ?? "Preparing the next component…"}</p>
                <p>Downloads and checksum verification can take several minutes. Existing files are checked and reused.</p>
              {/if}
              <small>Elapsed: {Math.floor(elapsedSeconds / 60)}m {elapsedSeconds % 60}s</small>
            {/if}
            {#if working || task >= 0}
              <ul class="tasks" aria-label="Preparation progress">{#each tasks as label, index}<li class:active={index === task} class:finished={index < task}><span>{index < task ? "✓" : index === task ? "•" : "○"}</span>{label}</li>{/each}</ul>
              {#if task === 1}{#each selectedDownloads.filter((entry) => entry.error) as download}<p class="error" role="alert">{download.error}</p>{/each}{/if}
            {/if}
          {:else if stage === "cleanup"}
            <h3>Remove the old SA3 Python installation</h3>
            <p class="passed">✓ Automatic runtime, generation{includeTraining ? ", training and checkpoint" : ""} checks passed.</p>
            <div class="summary">About {formatBytes(preview.estimatedCleanupBytes)} in reviewed Python files<small>Includes SA3 environments, PyTorch weights and unchanged bundled code. Actual recovered space may differ.</small></div>
            <p>Your original LoRAs, checkpoints, datasets, prompts, native models and other services are preserved. Edited or unrecognized Python source files are kept.</p>
            <p>This step stops SA3 and permanently removes the reviewed files. Gary verifies C++ again before cleanup and stops if the storage review has changed.</p>
            <details><summary>Files to remove ({preview.cleanupCandidates.length})</summary><div class="file-list">{#each preview.cleanupCandidates as entry}<div class="file"><span>{entry.label}</span><button class="path" type="button" onclick={() => onReveal(entry.path)}>{entry.path}</button></div>{/each}</div></details>
            <details><summary>Preserved files and folders</summary>{#each preview.preservedPaths as entry}<div class="file"><span>{entry.label}</span><button class="path" type="button" onclick={() => onReveal(entry.path)}>{entry.path}</button></div>{/each}<p>The shared UV cache stays available to your other Python services. It can be cleared separately in Storage.</p></details>
            {#each preview.warnings as warning}<p class="error" role="alert">{warning}</p>{/each}
            {#if preview.warnings.length}<p>Resolve these storage warnings, then check again before cleanup.</p><button type="button" onclick={initialize} disabled={blocked}>check again</button>{/if}
            {#each preview.nativeSelection?.cleanupErrors ?? [] as issue}<p class="error">{issue}</p>{/each}
          {:else}
            <h3>{preview.nativeSelection?.cleanupComplete ? "Migration complete" : "Ready to use SA3 C++"}</h3>
            <p>{preview.nativeSelection?.cleanupComplete ? "The reviewed Python files have been cleaned up. SA3 now uses the C++ runtime." : "C++ is selected and there are no reviewed Python files to clean up."}</p>
            <p>Your LoRAs, training history and prompts remain available.</p>
          {/if}
          <details class="storage"><summary>Using your current storage folder</summary><button class="path" type="button" onclick={() => onReveal(preview!.activeRoot)}>{preview.activeRoot}</button><p>PyTorch cache: {preview.hfHubCache}</p></details>
        {/if}
        {#if pendingRestart}<p class="error" role="alert">Restart Gary to use your chosen storage folder before migrating.</p>{/if}
        {#if working && stage !== "setup"}<p class="progress" role="status">{progress}</p>{/if}
        {#if error}<p class="error" role="alert">{error}</p>{#if !preview}<button type="button" onclick={initialize} disabled={working || loading}>check again</button>{/if}{/if}
      </div>
      <footer>
        {#if stage === "setup"}<button type="button" onclick={close} disabled={working || loading}>later</button><button class="primary" type="button" onclick={migrate} disabled={blocked || !preview}>{working ? "migrating…" : error && task >= 0 ? "retry migration" : preview?.nativeSelection ? "check and finish migration" : "prepare and switch to C++"}</button>
        {:else if stage === "cleanup"}<button type="button" onclick={close} disabled={working || loading}>keep Python for now</button><button class="primary" type="button" onclick={cleanup} disabled={blocked || !!preview?.warnings.length}>{working ? "cleaning up…" : "clean up Python and finish"}</button>
        {:else}<button class="primary" type="button" onclick={close}>done</button>{/if}
      </footer>
    </div>
  </div>
{/if}

<style>
  .overlay { position:fixed; inset:0; z-index:120; display:flex; align-items:center; justify-content:center; padding:24px; }
  .backdrop { position:absolute; inset:0; width:100%; height:100%; border:0; border-radius:0; background:rgba(0,0,0,.65); }
  .modal { position:relative; display:flex; flex-direction:column; width:580px; max-width:100%; max-height:calc(100vh - 48px); background:var(--bg-secondary,#202024); border:1px solid var(--border,#444); border-radius:12px; box-shadow:0 20px 70px #0008; color:var(--text-primary,#eee); outline:none; }
  header { display:flex; align-items:center; justify-content:space-between; padding:22px 24px 16px; }
  h2 { margin:5px 0 0; font-size:21px; font-weight:600; }
  h3 { margin:0 0 10px; font-size:16px; font-weight:600; }
  .eyebrow { color:var(--text-secondary,#aaa); font-size:11px; letter-spacing:1px; text-transform:uppercase; }
  button,select { border:1px solid var(--border,#444); border-radius:6px; color:inherit; background:var(--bg-panel,#25252a); padding:8px 12px; font:inherit; font-size:12px; }
  button { cursor:pointer; }
  button:disabled,select:disabled { opacity:.5; cursor:default; }
  .close { font-size:22px; line-height:1; padding:3px 8px; background:transparent; border:0; }
  .steps { display:flex; list-style:none; margin:0; padding:0 24px 18px; gap:14px; border-bottom:1px solid var(--border,#444); }
  .steps li { display:flex; align-items:center; gap:7px; font-size:11px; color:var(--text-secondary,#aaa); }
  .steps span { display:grid; place-items:center; border:1px solid var(--border,#444); border-radius:50%; width:22px; height:22px; }
  .steps .current { color:var(--text-primary,#eee); }
  .steps .current span { border-color:var(--accent,#547ec9); background:var(--accent,#547ec9); color:white; }
  .steps .done span { color:var(--green,#87c88c); }
  .content { padding:22px 24px; overflow-y:auto; min-height:0; }
  p { color:var(--text-secondary,#aaa); font-size:12px; line-height:1.6; margin:8px 0 14px; white-space:pre-wrap; overflow-wrap:anywhere; }
  .summary { background:var(--bg-panel,#29292e); border:1px solid var(--border,#444); border-radius:8px; padding:12px; margin:16px 0; font-size:13px; }
  small { display:block; margin-top:5px; color:var(--text-secondary,#aaa); font-size:11px; line-height:1.5; }
  details { margin-top:16px; font-size:12px; }
  summary { cursor:pointer; }
  .options { display:flex; flex-wrap:wrap; align-items:end; gap:12px; margin-top:12px; }
  label { font-size:12px; }
  select { display:block; margin-top:6px; }
  .tasks { list-style:none; padding:0; margin:18px 0 0; font-size:12px; }
  .tasks li { display:flex; gap:10px; padding:5px 0; color:var(--text-secondary,#aaa); }
  .tasks .active { color:var(--text-primary,#eee); }
  .tasks .finished span { color:var(--green,#87c88c); }
  .download-summary { display:flex; justify-content:space-between; gap:12px; margin:18px 0 10px; font-size:13px; }
  progress { width:100%; height:9px; accent-color:var(--accent,#547ec9); }
  .file-progress { color:var(--text-primary,#eee); }
  .passed { color:var(--green,#87c88c); }
  .progress { margin-top:16px; color:var(--text-primary,#eee); }
  .error { color:var(--red,#f99); }
  .storage { padding-top:14px; border-top:1px solid var(--border,#444); }
  .path { border:0; background:transparent; display:block; text-align:left; overflow-wrap:anywhere; padding:5px 0; font-size:11px; color:var(--text-secondary,#aaa); width:100%; }
  .path:hover { text-decoration:underline; }
  .file { margin-top:12px; font-size:12px; }
  .file-list { max-height:210px; overflow:auto; }
  footer { display:flex; justify-content:flex-end; gap:10px; padding:16px 24px; border-top:1px solid var(--border,#444); }
  .primary { background:var(--accent,#547ec9); border-color:var(--accent,#547ec9); color:white; }
  @media(max-width:600px) { .overlay { padding:12px; } .modal { max-height:calc(100vh - 24px); } .steps { gap:8px; } footer { flex-wrap:wrap; } }
</style>
