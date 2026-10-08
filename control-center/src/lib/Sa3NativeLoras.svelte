<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";

  interface NativeLora {
    name: string;
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

  let { pendingRestart = false, refreshKey = null }: { pendingRestart?: boolean; refreshKey?: object | null } = $props();
  let loraState: NativeLoraState | null = $state(null);
  let busy = $state(false);
  let mounted = $state(false);
  let error: string | null = $state(null);
  let message: string | null = $state(null);
  let disposed = false;
  let selected: Record<string, string> = $state({});

  async function refresh() {
    const result = await invoke<NativeLoraState>("get_sa3_native_lora_state");
    if (!disposed) loraState = result;
  }

  $effect(() => {
    refreshKey;
    if (mounted) void refresh().catch((cause) => { if (!disposed) error = String(cause); });
  });

  onMount(() => {
    mounted = true;
    const listener = listen<NativeLoraState>("sa3-native-loras-updated", (event) => {
      if (!disposed) loraState = event.payload;
    });
    return () => { disposed = true; void listener.then((stop) => stop()); };
  });

  async function prepare() {
    busy = true;
    error = null;
    message = null;
    try {
      loraState = await invoke<NativeLoraState>("prepare_sa3_native_loras");
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
      loraState = await invoke<NativeLoraState>("select_sa3_native_checkpoint", { name: entry.name, checkpoint });
      message = `Selected checkpoint for ${entry.name}. Training originals are preserved.`;
    } catch (cause) { error = String(cause); }
    finally { busy = false; }
  }
</script>

<section aria-label="SA3 native adapters">
  <h3>Prepare LoRAs for C++</h3>
  <p>Convert registered adapters into native copies in your selected storage folder. Prepare the C++ runtime first, and stop native SA3 before conversion.</p>
  {#if loraState?.entries.length}
    <button type="button" onclick={prepare} disabled={busy || loraState.preparing || pendingRestart}>
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
              <select bind:value={selected[entry.name]} disabled={busy || loraState.preparing || pendingRestart}>
                <option value="">Choose a checkpoint</option>
                {#each entry.trainingCheckpoints as checkpoint (checkpoint.path)}
                  <option value={checkpoint.path}>step {checkpoint.step}{checkpoint.epoch === null ? "" : ` · epoch ${checkpoint.epoch}`} · {checkpoint.jobId}{checkpoint.path === entry.sourcePath ? " · selected" : ""}</option>
                {/each}
              </select>
            </label>
            <button type="button" onclick={() => selectCheckpoint(entry)} disabled={busy || loraState.preparing || pendingRestart || !selected[entry.name]}>use checkpoint</button>
          </div>
        {/if}
        {#if entry.promptsPath}<p>Caption prompts: {entry.promptsPath}</p>{/if}
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
</style>
