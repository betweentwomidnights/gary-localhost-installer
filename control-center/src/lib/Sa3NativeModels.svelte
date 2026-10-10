<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";
  import { recommendQuantizedSa3Training, type Sa3TrainingHardware } from "./sa3NativeTraining";

  interface Component { id: string; label: string; description: string; files: { bytes: number }[]; }
  interface Model { id: string; status: "available" | "downloading" | "downloaded" | "failed"; downloaded_bytes: number | null; }
  interface Progress { model_id: string; progress: number; status: string; message: string; error: string | null; }
  interface Decoder { enabled: boolean; sourcePresent: boolean; prepared: boolean; error: string | null; }
  let { models, progress, nativeSelected, onDownload, onRemove, removingId }: {
    models: Model[];
    progress: Map<string, Progress>;
    nativeSelected: boolean;
    onDownload: (id: string) => void;
    onRemove: (id: string) => void;
    removingId: string | null;
  } = $props();
  let catalog = $state<Component[]>([]);
  let activeEncoding = $state<string | null>(null);
  let trainingRecommendation = $state("F16");
  let decoder = $state<Decoder | null>(null);
  let busy = $state(false);
  let error = $state<string | null>(null);
  let message = $state<string | null>(null);
  let sharedReady = $derived(["sa3-native::text", "sa3-native::medium-decoder"].every((id) => models.find((model) => model.id === id)?.status === "downloaded"));
  let downloading = $derived(models.some((model) => model.id.startsWith("sa3-native::") && model.status === "downloading"));
  let fix = $derived(models.find((model) => model.id === "sa3-native::decoder-correction"));
  let fixProgress = $derived(progress.get("sa3-native::decoder-correction"));
  const groups = [
    { label: "shared components", kind: "shared", hint: "Download these GGUF components once for generation and LoRA training. Text encoder: F16. Autoencoder: F32. The tokenizer and conditioner are included." },
    { label: "generation tier", kind: "generation", hint: "One tier is enough. F16 is the default; smaller DiT tiers reduce memory use. Shared component precision stays the same." },
    { label: "LoRA training base", kind: "training", hint: "Optional — download one base only if you train LoRAs. Training precision is independent of the generation tier." },
  ];
  function kind(id: string) {
    if (["sa3-native::text", "sa3-native::medium-decoder"].includes(id)) return "shared";
    return id.includes("medium-base-") ? "training" : "generation";
  }
  function encoding(id: string) { return id.replace(/^sa3-native::medium-(?:base-)?/, ""); }
  function size(bytes: number) { return `${(bytes / 1024 ** 3).toFixed(2)} GB`; }
  function label(entry: Component) {
    if (entry.id === "sa3-native::text") return "Text encoder and tokenizer · F16";
    if (entry.id === "sa3-native::medium-decoder") return "Autoencoder and conditioner · F32";
    return `${entry.id.includes("medium-base-") ? "Medium training base" : "Medium"} · ${encoding(entry.id)}`;
  }
  async function loadState() {
    const [selection, state] = await Promise.all([
      invoke<{ encoding: string } | null>("get_sa3_native_runtime_selection"),
      invoke<Decoder>("get_sa3_native_decoder_state"),
    ]);
    activeEncoding = selection?.encoding ?? null;
    decoder = state;
  }
  $effect(() => { void nativeSelected; void loadState().catch((cause) => error = String(cause)); });
  onMount(() => {
    let disposed = false;
    void invoke<Component[]>("get_sa3_native_model_catalog").then((entries) => { if (!disposed) catalog = entries; }).catch((cause) => { if (!disposed) error = String(cause); });
    void invoke<Sa3TrainingHardware>("get_native_runtime_info", { serviceId: "sa3" }).then((info) => {
      if (!disposed && recommendQuantizedSa3Training(info)) trainingRecommendation = "Q4_K_M";
    }).catch((cause) => { if (!disposed) error = String(cause); });
    const listeners = [
      listen<{ encoding: string }>("sa3-native-model-selected", (event) => { if (!disposed) activeEncoding = event.payload.encoding; }),
      listen<Decoder>("sa3-native-decoder-updated", (event) => { if (!disposed) decoder = event.payload; }),
    ];
    return () => { disposed = true; for (const listener of listeners) void listener.then((stop) => stop()); };
  });
  async function useTier(tier: string) {
    busy = true; error = null; message = null;
    try {
      const selection = await invoke<{ encoding: string }>("select_sa3_native_model", { encoding: tier });
      activeEncoding = selection.encoding;
      message = `SA3 will use ${tier} next time it starts.`;
    } catch (cause) { error = String(cause); }
    finally { busy = false; }
  }
  async function convertDecoder() {
    busy = true; error = null; message = null;
    try {
      decoder = await invoke<Decoder>("prepare_sa3_native_decoder");
      message = "Decoder fix installed. Enable it with the decoder squeak fix switch.";
    } catch (cause) { error = String(cause); }
    finally { busy = false; }
  }
</script>

{#each groups as group}
  <section aria-label={group.label}>
    <div class="group-label">{group.label}</div>
    <p>{group.hint}
      {#if group.kind === "generation" && activeEncoding}SA3 is set to {activeEncoding}. Stop SA3 before changing its model.{/if}
      {#if group.kind === "training"}{trainingRecommendation} is recommended for training on this hardware.{/if}
    </p>
    {#each catalog.filter((entry) => entry.id !== "sa3-native::decoder-correction" && kind(entry.id) === group.kind) as entry (entry.id)}
      {@const model = models.find((item) => item.id === entry.id)}
      {@const download = progress.get(entry.id)}
      <div class="model-row" class:downloaded={model?.status === "downloaded"}>
        <div class="model-info">
          <span class="model-name">{label(entry)}
            {#if group.kind === "generation" && activeEncoding === encoding(entry.id)}<span class="badge active">active</span>
            {:else if group.kind === "generation" && encoding(entry.id) === "F16"}<span class="badge">default</span>{/if}
            {#if group.kind === "training" && encoding(entry.id) === trainingRecommendation}<span class="badge">recommended</span>{/if}
          </span>
          <span class="model-size">{size(entry.files.reduce((sum, file) => sum + file.bytes, 0))}{model?.status === "downloaded" ? " on disk" : " download"}</span>
          {#if model?.status === "downloading"}<span class="model-size" role="status">{download?.message ?? "Queued for download"}</span>{/if}
          {#if download?.error}<span class="error" role="alert">{download.error}</span>{/if}
        </div>
        {#if model?.status === "downloading"}
          <div class="download-progress"><progress max="1" value={download?.progress ?? 0} aria-label={`${label(entry)} download`}></progress><span>{Math.round((download?.progress ?? 0) * 100)}%</span></div>
        {:else if model?.status === "downloaded"}
          <div class="actions">
            {#if group.kind === "generation" && nativeSelected && activeEncoding !== encoding(entry.id)}<button class="download" disabled={busy || downloading || removingId !== null || !sharedReady} title={sharedReady ? "Use this tier next time SA3 starts" : "Download the shared components first"} onclick={() => useTier(encoding(entry.id))}>use</button>{/if}
            <button disabled={busy || downloading || removingId !== null} onclick={() => onRemove(entry.id)}>{removingId === entry.id ? "removing..." : "remove"}</button>
          </div>
        {:else}<button class="download" disabled={busy || downloading || removingId !== null} onclick={() => onDownload(entry.id)}>{model?.status === "failed" ? "Retry" : "Download"}</button>{/if}
      </div>
    {/each}
  </section>
{/each}
<section aria-label="optional decoder fix">
  <div class="group-label">optional decoder fix</div>
  <p>A small decoder LoRA, converted for C++ without Python. After installation, enable the decoder squeak fix switch.</p>
  <div class="model-row">
    <div class="model-info"><span class="model-name">Decoder squeak fix {#if decoder?.prepared}<span class="badge active">installed</span>{/if}</span>
      {#if fix?.status === "downloading"}<span class="model-size" role="status">{fixProgress?.message ?? "Downloading decoder fix…"}</span>{/if}
      {#if fixProgress?.error || decoder?.error}<span class="error">{fixProgress?.error ?? decoder?.error}</span>{/if}
    </div>
    {#if fix?.status === "downloading"}<div class="download-progress"><progress max="1" value={fixProgress?.progress ?? 0} aria-label="Decoder fix download"></progress><span>{Math.round((fixProgress?.progress ?? 0) * 100)}%</span></div>
    {:else if !decoder?.prepared}
      {#if fix?.status === "downloaded" || decoder?.sourcePresent}<button class="download" disabled={busy || downloading || removingId !== null} onclick={convertDecoder}>install decoder fix</button>
      {:else}<button class="download" disabled={busy || downloading || removingId !== null} onclick={() => onDownload("sa3-native::decoder-correction")}>{fix?.status === "failed" ? "Retry" : "Download"}</button>{/if}
    {/if}
  </div>
</section>
{#if !nativeSelected}<p>Use migrate runtime on the SA3 service row to switch to C++ after downloading.</p>{/if}
{#if message}<p role="status">{message}</p>{/if}
{#if error}<p class="error" role="alert">{error}</p>{/if}

<style>
  section { margin-bottom: 12px; }
  .group-label { padding: 4px 16px; font-size: 10px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px; }
  p { margin: 4px 16px 10px; font-size: 11px; color: var(--text-secondary); line-height: 1.5; }
  .model-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 6px 16px; transition: background 0.1s; }
  .model-row:hover { background: var(--bg-hover); }
  .model-row.downloaded { opacity: 0.7; }
  .model-info { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
  .model-name { font-size: 12px; color: var(--text-primary); }
  .model-size { font-size: 10px; color: var(--text-muted); overflow-wrap: anywhere; }
  .actions { display: flex; gap: 6px; }
  button { flex-shrink: 0; font-size: 11px; }
  button.download { color: var(--accent); border-color: var(--accent); }
  .badge { display: inline-block; margin-left: 5px; padding: 1px 4px; border: 1px solid var(--border); border-radius: 3px; font-size: 9px; color: var(--text-muted); }
  .badge.active { color: var(--green); border-color: var(--green); }
  .download-progress { display: flex; align-items: center; gap: 6px; font-size: 10px; color: var(--text-secondary); }
  progress { width: 80px; height: 6px; }
  .error { color: var(--red); font-size: 11px; overflow-wrap: anywhere; }
</style>
