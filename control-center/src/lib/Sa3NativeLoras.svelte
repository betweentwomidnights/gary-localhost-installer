<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";
  import { open as openDialog } from "@tauri-apps/plugin-dialog";
  import { rememberedDialogDirectory, rememberDialogSelection } from "./dialogMemory";

  interface NativeLora {
    name: string;
    nativeOnly: boolean;
    strength: number;
    sourcePath: string;
    nativePath: string | null;
    error: string | null;
    promptsPath: string | null;
    trainingCheckpoints: { jobId: string; step: number; epoch: number | null; path: string; statePath: string }[];
  }
  interface NativeLoraState {
    entries: NativeLora[];
    preparing: boolean;
    adaptersDir: string;
  }

  let { pendingRestart = false, refreshKey = null, allowImport = false }: { pendingRestart?: boolean; refreshKey?: object | null; allowImport?: boolean } = $props();
  let loraState: NativeLoraState | null = $state(null);
  let busy = $state(false);
  let mounted = $state(false);
  let error: string | null = $state(null);
  let message: string | null = $state(null);
  let disposed = false;
  let selected: Record<string, string> = $state({});
  let strengths: Record<string, number> = $state({});
  let nativeRunning = $state(false);
  let formName = $state("");
  let sourcePath = $state("");
  let configPath = $state("");
  let datasetPath = $state("");
  const blocked = $derived.by(() => busy || !!loraState?.preparing || pendingRestart || nativeRunning);

  function applyState(state: NativeLoraState) {
    loraState = state;
    strengths = Object.fromEntries(state.entries.map((entry) => [entry.name, entry.strength]));
  }
  function applyServices(services: {id:string;runtime:string;status:string}[]) {
    nativeRunning = services.some((service) => service.id === "sa3" && service.runtime === "native" && ["starting","running","unhealthy"].includes(service.status));
  }

  async function refresh() {
    const result = await invoke<NativeLoraState>("get_sa3_native_lora_state");
    if (!disposed) applyState(result);
  }

  $effect(() => {
    refreshKey;
    if (mounted) void refresh().catch((cause) => { if (!disposed) error = String(cause); });
  });

  onMount(() => {
    mounted = true;
    const listener = listen<NativeLoraState>("sa3-native-loras-updated", (event) => {
      if (!disposed) applyState(event.payload);
    });
    const servicesListener = listen<{id:string;runtime:string;status:string}[]>("services-updated", (event) => { if (!disposed) applyServices(event.payload); });
    void invoke<{id:string;runtime:string;status:string}[]>("get_services").then((services) => { if (!disposed) applyServices(services); }).catch((cause) => error = String(cause));
    return () => { disposed = true; void listener.then((stop) => stop()); void servicesListener.then((stop) => stop()); };
  });

  async function prepare() {
    busy = true;
    error = null;
    message = null;
    try {
      applyState(await invoke<NativeLoraState>("prepare_sa3_native_loras"));
      if (!loraState) return;
      const prepared = loraState.entries.filter((entry) => entry.nativePath && !entry.error).length;
      message = `${prepared} of ${loraState.entries.length} adapters prepared. Originals are preserved.`;
    } catch (cause) {
      error = String(cause);
    } finally {
      busy = false;
      await refresh().catch((cause) => { error ??= String(cause); });
    }
  }

  async function reveal(path: string) {
    try { await invoke("reveal_path", { path }); }
    catch (cause) { error = String(cause); }
  }

  async function selectCheckpoint(entry: NativeLora) {
    const checkpoint = selected[entry.name];
    if (!checkpoint) return;
    busy = true;
    error = null;
    message = null;
    try {
      applyState(await invoke<NativeLoraState>("select_sa3_native_checkpoint", { name: entry.name, checkpoint }));
      message = `Selected checkpoint for ${entry.name}. Training originals are preserved.`;
    } catch (cause) { error = String(cause); }
    finally { busy = false; }
  }

  async function pickFile(field: "source" | "config") {
    const key = `sa3-native-lora-${field}`;
    const selected = await openDialog({ directory:false, multiple:false, defaultPath:rememberedDialogDirectory(key), filters:[{ name:field === "source" ? "SA3 LoRA" : "LoRA configuration", extensions:field === "source" ? ["gguf","safetensors","ckpt"] : ["json"] }] });
    if (typeof selected !== "string") return;
    rememberDialogSelection(key,selected,"file");
    if (field === "config") configPath = selected;
    else {
      sourcePath = selected;
      if (!formName.trim()) formName = selected.replace(/\\/g,"/").split("/").pop()!.replace(/\.(gguf|safetensors|ckpt)$/i,"").toLowerCase().replace(/[^a-z0-9_-]+/g,"-").replace(/^-+|-+$/g,"").slice(0,64);
    }
  }
  async function pickDataset() {
    const selected = await openDialog({ directory:true, multiple:false, defaultPath:rememberedDialogDirectory("sa3-lora-prompts") });
    if (typeof selected === "string") { rememberDialogSelection("sa3-lora-prompts",selected,"directory"); datasetPath = selected; }
  }
  async function importAdapter() {
    busy = true; error = null; message = null;
    try {
      applyState(await invoke<NativeLoraState>("import_sa3_native_lora", { name:formName, checkpointPath:sourcePath, configPath:/\.gguf$/i.test(sourcePath.trim()) ? null : configPath.trim() || null, promptsPath:datasetPath.trim() || null }));
      message = `Imported ${formName.trim()}. The source file is preserved.`;
      formName = ""; sourcePath = ""; configPath = ""; datasetPath = "";
    } catch (cause) { error = String(cause); }
    finally { busy = false; }
  }
  async function update(entry: NativeLora, change: {action:"strength";value:number} | {action:"remove" | "build_prompts"}) {
    busy = true; error = null; message = null;
    try {
      applyState(await invoke<NativeLoraState>("update_sa3_native_lora", { name:entry.name, change }));
      message = change.action === "remove" ? `Unregistered ${entry.name}. Source files, prepared copies, prompts and checkpoints are preserved.` : change.action === "build_prompts" ? `Rebuilt caption prompts for ${entry.name}.` : `Saved suggested strength for ${entry.name}.`;
    } catch (cause) { error = String(cause); }
    finally { busy = false; }
  }
</script>

<section aria-label="SA3 native adapters">
  <h3>{allowImport ? "Native LoRAs" : "Prepare LoRAs for C++"}</h3>
  <p>Convert registered adapters into native copies in your selected storage folder. Prepare the C++ runtime first, and stop native SA3 before conversion.</p>
  {#if allowImport}
    <div class="import-form">
      <label>Name<input bind:value={formName} disabled={blocked} placeholder="my-lora" /></label>
      <label>Adapter file<div class="file-row"><input bind:value={sourcePath} disabled={blocked} placeholder="GGUF, safetensors or CKPT" /><button type="button" disabled={blocked} onclick={() => pickFile("source")}>pick file</button></div></label>
      {#if !/\.gguf$/i.test(sourcePath.trim())}
        <label>LoRA JSON configuration (optional)<div class="file-row"><input bind:value={configPath} disabled={blocked} placeholder="Embedded metadata or matching sidecar is used automatically" /><button type="button" disabled={blocked} onclick={() => pickFile("config")}>pick JSON</button></div></label>
      {/if}
      <label>Caption dataset (optional)<div class="file-row"><input bind:value={datasetPath} disabled={blocked} /><button type="button" disabled={blocked} onclick={pickDataset}>pick folder</button></div></label>
      <button type="button" disabled={blocked || !formName.trim() || !sourcePath.trim()} onclick={importAdapter}>import native LoRA</button>
    </div>
  {/if}
  {#if nativeRunning}<p>Stop SA3 before importing, preparing or changing native adapters.</p>{/if}
  {#if loraState?.entries.length}
    <button type="button" onclick={prepare} disabled={blocked}>
      {busy || loraState.preparing ? "preparing adapters..." : "prepare / verify adapters"}
    </button>
    {#each loraState.entries as entry (entry.name)}
      <div class="entry">
        <div class="row"><span>{entry.name}</span><span>{entry.nativePath && !entry.error ? "prepared" : "needs preparation"}</span></div>
        <button type="button" class="path" onclick={() => reveal(entry.sourcePath)} title="Show original adapter">{entry.sourcePath}</button>
        {#if entry.nativePath}<button type="button" class="path" onclick={() => entry.nativePath && reveal(entry.nativePath)} title="Show native copy">{entry.nativePath}</button>{/if}
        {#if entry.trainingCheckpoints.length}
          <div class="checkpoints">
            <label>Training checkpoint
              <select bind:value={selected[entry.name]} disabled={blocked}>
                <option value="">Choose a checkpoint</option>
                {#each entry.trainingCheckpoints as checkpoint (checkpoint.path)}
                  <option value={checkpoint.path}>step {checkpoint.step}{checkpoint.epoch === null ? "" : ` · epoch ${checkpoint.epoch}`} · {checkpoint.jobId}{checkpoint.path === entry.sourcePath ? " · selected" : ""}</option>
                {/each}
              </select>
            </label>
            <button type="button" onclick={() => selectCheckpoint(entry)} disabled={blocked || !selected[entry.name]}>use checkpoint</button>
          </div>
        {/if}
        {#if entry.promptsPath}<p>Caption prompts: {entry.promptsPath}</p>{/if}
        <div class="entry-controls">
          <label>Suggested strength<input type="number" min="-4" max="4" step="0.05" bind:value={strengths[entry.name]} disabled={blocked} /></label>
          <button type="button" disabled={blocked || !Number.isFinite(strengths[entry.name])} onclick={() => update(entry,{action:"strength",value:strengths[entry.name]})}>save strength</button>
          {#if entry.promptsPath}<button type="button" title="Replaces this adapter's prompt pool with prompts from its caption dataset" disabled={blocked} onclick={() => update(entry,{action:"build_prompts"})}>rebuild prompts</button>{/if}
          {#if entry.nativeOnly || /\.gguf$/i.test(entry.sourcePath)}<button type="button" disabled={blocked} onclick={() => update(entry,{action:"remove"})}>unregister</button>{/if}
        </div>
        {#if entry.error}<p class="error">{entry.error}</p>{/if}
      </div>
    {/each}
  {:else if loraState}
    <p>No registered LoRAs to prepare. Add an exported adapter in the SA3 LoRA manager.</p>
  {/if}
  <p>Legacy .ckpt adapters are exported using your existing SA3 Python environment. Prepare them before cleanup; cached copies remain usable afterward. Originals and training history are preserved.</p>
  {#if message}<p role="status">{message}</p>{/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
</section>

<style>
  section { margin: 18px 0; padding: 14px; border: 1px solid var(--border, #333); border-radius: 8px; }
  h3 { margin: 0; font-size: 14px; font-weight: 600; }
  p { margin: 6px 0 10px; font-size: 12px; color: var(--text-secondary, #aaa); line-height: 1.5; }
  button { background: var(--bg-secondary, #222); border: 1px solid var(--border, #444); color: inherit; border-radius: 6px; padding: 7px 10px; font: inherit; font-size: 12px; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
  .entry { margin-top: 12px; font-size: 12px; }
  .row { display: flex; justify-content: space-between; gap: 12px; }
  .checkpoints { display: flex; align-items: end; gap: 8px; margin-top: 8px; }
  label { flex: 1; min-width: 0; }
  select { display: block; width: 100%; margin-top: 4px; background: var(--bg-secondary, #222); border: 1px solid var(--border, #444); color: inherit; padding: 6px; border-radius: 6px; }
  .path { display: block; text-align: left; border: none; background: transparent; padding: 4px 0; color: var(--text-secondary, #aaa); overflow-wrap: anywhere; width: 100%; }
  .path:hover { text-decoration: underline; }
  .error { color: var(--error, #f99); overflow-wrap: anywhere; }
  .import-form { display:grid; gap:10px; margin:12px 0; }
  .file-row, .entry-controls { display:flex; gap:8px; align-items:end; flex-wrap:wrap; }
  .file-row input { flex:1; min-width:180px; }
  .entry-controls { margin-top:10px; }
  .entry-controls label { flex:0 0 130px; }
  input { display:block; width:100%; box-sizing:border-box; margin-top:4px; padding:6px; background:var(--bg-secondary,#222); border:1px solid var(--border,#444); color:inherit; border-radius:6px; font:inherit; font-size:12px; }
</style>
