<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";
  import { recommendQuantizedSa3Training, type Sa3TrainingHardware } from "./sa3NativeTraining";

  interface Component {
    id: string;
    label: string;
    description: string;
    files: { bytes: number }[];
  }
  interface Model {
    id: string;
    status: "available" | "downloading" | "downloaded" | "failed";
  }
  interface Progress {
    model_id: string;
    progress: number;
    status: string;
    message: string;
    error: string | null;
  }
  interface DecoderState { enabled: boolean; sourcePresent: boolean; prepared: boolean; error: string | null; }

  let { pendingRestart = false }: { pendingRestart?: boolean } = $props();
  let catalog: Component[] = $state([]);
  let models: Model[] = $state([]);
  let progress: Progress[] = $state([]);
  let encoding = $state("F16");
  let includeTraining = $state(true);
  let includeDecoder = $state(false);
  let decoderTouched = $state(false);
  let decoder: DecoderState | null = $state(null);
  let trainingBase = $state("F16");
  let trainingBaseTouched = $state(false);
  let busy = $state(false);
  let removing: string | null = $state(null);
  let error: string | null = $state(null);
  let message: string | null = $state(null);

  const selectedIds = $derived([
    "sa3-native::text",
    "sa3-native::medium-decoder",
    `sa3-native::medium-${encoding}`,
    ...(includeTraining ? [`sa3-native::medium-base-${trainingBase}`] : []),
    ...(includeDecoder ? ["sa3-native::decoder-correction"] : []),
  ]);
  const selected = $derived(catalog.filter((entry) => selectedIds.includes(entry.id)));
  const downloading = $derived(models.some((model) => model.id.startsWith("sa3-native::") && model.status === "downloading"));
  const totalBytes = $derived(selected.reduce((total, entry) => total + entry.files.reduce((bytes, file) => bytes + file.bytes, 0), 0));
  const ready = $derived(selected.length > 0 && selected.every((entry) => models.find((model) => model.id === entry.id)?.status === "downloaded"));

  function formatBytes(bytes: number) {
    return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  }

  onMount(() => {
    let disposed = false;
    const listeners = [
      listen<Model[]>("models-updated", (event) => { if (!disposed) { models = event.payload; void refreshDecoder(); } }),
      listen<Progress[]>("download-progress", (event) => { if (!disposed) progress = event.payload; }),
      listen<DecoderState>("sa3-native-decoder-updated", (event) => { if (!disposed) decoder = event.payload; }),
    ];
    async function refreshDecoder() {
      try {
        const state = await invoke<DecoderState>("get_sa3_native_decoder_state");
        if (!disposed) { decoder = state; if (!decoderTouched && state.enabled) includeDecoder = true; }
      } catch (cause) { if (!disposed) error = String(cause); }
    }
    void refreshDecoder();
    void Promise.all([
      invoke<Component[]>("get_sa3_native_model_catalog"),
      invoke<Model[]>("get_models"),
      invoke<Progress[]>("get_download_progress"),
    ]).then(([entries, statuses, downloads]) => {
      if (disposed) return;
      catalog = entries;
      models = statuses;
      progress = downloads;
    }).catch((cause) => { if (!disposed) error = String(cause); });
    void invoke<Sa3TrainingHardware>("get_native_runtime_info", { serviceId: "sa3" }).then((info) => {
      if (!disposed && !trainingBaseTouched && recommendQuantizedSa3Training(info)) trainingBase = "Q4_K_M";
    }).catch((cause) => { if (!disposed) error = String(cause); });
    return () => { disposed = true; for (const listener of listeners) void listener.then((stop) => stop()); };
  });

  async function prepare() {
    busy = true;
    error = null;
    message = null;
    try {
      await invoke("prepare_sa3_native_models", { encoding, trainingBase: includeTraining ? trainingBase : null, includeDecoder });
      message = "Preparing your selected model components. Existing files are verified and reused.";
      models = await invoke<Model[]>("get_models");
    } catch (cause) {
      error = String(cause);
    } finally {
      busy = false;
    }
  }

  async function prepareDecoder() {
    busy = true; error = null; message = null;
    try {
      decoder = await invoke<DecoderState>("prepare_sa3_native_decoder");
      message = "Native decoder correction is prepared. Enable it with the decoder squeak fix switch.";
    } catch (cause) { error = String(cause); }
    finally { busy = false; }
  }

  async function remove(entry: Component) {
    if (!window.confirm(`Remove ${entry.label} from this storage folder?\n\nSA3 must be stopped. You can download it again later.`)) return;
    removing = entry.id;
    error = null;
    message = null;
    try {
      await invoke("remove_model", { modelId: entry.id, serviceId: "sa3" });
      models = await invoke<Model[]>("get_models");
      message = `Removed ${entry.label}.`;
    } catch (cause) {
      error = String(cause);
    } finally {
      removing = null;
    }
  }
</script>

<section aria-labelledby="sa3-native-models-title">
  <h3 id="sa3-native-models-title">Prepare SA3 C++ models</h3>
  <p>Medium models are downloaded into your selected storage folder. The text encoder, tokenizer, decoder and conditioner are shared between inference tiers and training.</p>
  <div class="options">
    <label>Inference precision
      <select bind:value={encoding} disabled={busy || downloading || pendingRestart}>
        <option value="F16">F16 — default</option>
        <option value="Q8_0">Q8_0</option>
        <option value="Q5_K_M">Q5_K_M</option>
        <option value="Q4_K_M">Q4_K_M — smallest DiT</option>
      </select>
    </label>
    <label class="checkbox"><input type="checkbox" bind:checked={includeTraining} disabled={busy || downloading || pendingRestart} /> Include LoRA training base</label>
    <label class="checkbox"><input type="checkbox" bind:checked={includeDecoder} onchange={() => decoderTouched = true} disabled={busy || downloading || pendingRestart || decoder?.enabled} /> Include optional decoder squeak fix</label>
    {#if includeTraining}
      <label>Training precision
        <select bind:value={trainingBase} disabled={busy || downloading || pendingRestart} onchange={() => trainingBaseTouched = true}>
          <option value="F16">F16 — default</option>
          <option value="Q4_K_M">Q4_K_M — smaller base</option>
        </select>
      </label>
    {/if}
  </div>
  <p>Decoder: F32. Text encoder: F16. These stay at the same precision when you choose a smaller inference DiT.</p>
  {#if selected.length}
    <p>{formatBytes(totalBytes)} for this selection. Components already present are reused.</p>
    <button type="button" onclick={prepare} disabled={busy || downloading || removing !== null || pendingRestart}>
      {busy || downloading ? "preparing models..." : ready ? "verify prepared models" : "prepare models"}
    </button>
  {/if}
  {#each selected as entry (entry.id)}
    {@const model = models.find((model) => model.id === entry.id)}
    {@const download = progress.find((item) => item.model_id === entry.id)}
    <div class="component">
      <div class="row"><span>{entry.label}</span><span>{model?.status === "downloaded" ? "prepared" : model?.status ?? "available"}</span></div>
      {#if model?.status === "downloading"}
        <progress max="1" value={download?.progress ?? 0} aria-label={`${entry.label} download`}></progress>
        <p role="status">{download?.message ?? "Queued for preparation"}</p>
      {:else if model?.status === "downloaded"}
        <button type="button" onclick={() => remove(entry)} disabled={downloading || busy || removing !== null || pendingRestart}>{removing === entry.id ? "removing..." : "remove"}</button>
      {/if}
      {#if download?.error}<p class="error" role="alert">{download.error}</p>{/if}
    </div>
  {/each}
  {#if ready}<p role="status">Selected components are present. Preparing verifies their checksums again.</p>{/if}
  {#if includeDecoder || decoder?.sourcePresent || decoder?.enabled}
    <div class="component">
      <p>Decoder squeak fix: {decoder?.prepared ? "native copy prepared" : "native copy needs preparation"}. Stop SA3 before converting the source. Your creative LoRAs remain separately selectable.</p>
      <button type="button" onclick={prepareDecoder} disabled={busy || downloading || removing !== null || pendingRestart || !decoder?.sourcePresent}>{decoder?.prepared ? "verify native decoder fix" : "prepare native decoder fix"}</button>
      {#if decoder?.error}<p>{decoder.error}</p>{/if}
    </div>
  {/if}
  <p>Model preparation uses your selected storage folder. Verify and switch runtimes in Storage after preparing the components and adapters.</p>
  {#if message}<p role="status">{message}</p>{/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
</section>

<style>
  section { margin: 18px 0; padding: 14px; border: 1px solid var(--border, #333); border-radius: 8px; }
  h3 { margin: 0; font-size: 14px; font-weight: 600; }
  p { margin: 6px 0 10px; font-size: 12px; color: var(--text-secondary, #aaa); line-height: 1.5; }
  .options { display: flex; align-items: end; flex-wrap: wrap; gap: 12px; margin: 12px 0; }
  label { display: flex; flex-direction: column; gap: 6px; font-size: 12px; }
  .checkbox { flex-direction: row; align-items: center; padding-bottom: 6px; }
  button, select { background: var(--bg-secondary, #222); border: 1px solid var(--border, #444); color: inherit; border-radius: 6px; padding: 7px 10px; font: inherit; font-size: 12px; }
  button { cursor: pointer; }
  button:disabled, select:disabled { opacity: 0.5; cursor: default; }
  .component { margin-top: 12px; font-size: 12px; }
  .row { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 6px; }
  progress { width: 100%; height: 6px; }
  .error { color: var(--error, #f99); overflow-wrap: anywhere; }
</style>
