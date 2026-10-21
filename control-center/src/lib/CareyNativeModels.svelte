<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import DownloadProgress from "./DownloadProgress.svelte";
  import { onMount } from "svelte";

  interface Component { id: string; label: string; description: string; files: { bytes: number }[]; }
  interface Model { id: string; status: "available" | "downloading" | "downloaded" | "failed"; downloaded_bytes: number | null; }
  interface Progress { model_id: string; progress: number; status: string; message: string; error: string | null; }
  interface Recommendations { tier: string | null; components: string[]; }
  let { models, progress, nativeSelected, onDownload, onRemove, removingId }: {
    models: Model[];
    progress: Map<string, Progress>;
    nativeSelected: boolean;
    onDownload: (id: string) => void;
    onRemove: (id: string) => void;
    removingId: string | null;
  } = $props();

  // ace-server keeps every downloaded model in one registry and gary4juce names
  // the one it wants, so each model is its own choice of precision. The base
  // model at the active precision is the one Carey needs to start.
  const groups = [
    { label: "2B models", hint: "The base model runs every task, including lego, extract and complete, and Carey needs it to start. sft and turbo are optional.", variants: ["base", "sft", "turbo"] },
    { label: "XL models", hint: "Optional. XL is larger and needs more GPU memory than 2B at the same precision. Some tasks, such as lego, can sound better on 2B.", variants: ["xl-base", "xl-sft", "xl-turbo"] },
    { label: "language models", hint: "Optional. A language model writes lyrics, metadata and audio codes for standard generation; Carey's DAW tasks run without one.", variants: ["lm-0.6B", "lm-1.7B", "lm-4B"] },
  ];
  let catalog = $state<Component[]>([]);
  let recommended = $state<Recommendations>({ tier: null, components: [] });
  let activeEncoding = $state<string | null>(null);
  // The precision each row shows, by variant, once picked by hand.
  let picked = $state<Record<string, string>>({});
  let busy = $state(false);
  let error = $state<string | null>(null);
  let message = $state<string | null>(null);

  const shared = $derived(catalog.find((entry) => entry.id === "carey-native::shared"));
  const status = (id: string) => models.find((model) => model.id === id)?.status;
  const downloading = $derived(models.some((model) => model.id.startsWith("carey-native::") && model.status === "downloading"));
  const precisions = (variant: string) => catalog.filter((entry) => entry.id.startsWith(`carey-native::${variant}::`)).map((entry) => entry.id.split("::")[2]);
  // A row shows what was picked, else what is active or on disk, else the recommendation.
  function shown(variant: string) {
    const options = precisions(variant);
    const id = (tier: string) => `carey-native::${variant}::${tier}`;
    return picked[variant]
      ?? (variant === "base" && activeEncoding && options.includes(activeEncoding) ? activeEncoding : undefined)
      ?? options.find((tier) => status(id(tier)) === "downloaded")
      ?? options.find((tier) => recommended.components.includes(id(tier)))
      ?? options[0];
  }
  const onDisk = (variant: string) => precisions(variant).filter((tier) => status(`carey-native::${variant}::${tier}`) === "downloaded");
  function size(bytes: number) { return `${(bytes / 1024 ** 3).toFixed(2)} GB`; }
  function bytesOf(entry: Component) { return entry.files.reduce((sum, file) => sum + file.bytes, 0); }

  async function refresh() {
    const [selection, recommendations] = await Promise.all([
      invoke<{ encoding: string } | null>("get_native_runtime_selection", { serviceId: "carey" }),
      invoke<Recommendations>("get_native_model_recommendations", { serviceId: "carey" }),
    ]);
    activeEncoding = selection?.encoding ?? null;
    recommended = recommendations;
  }
  $effect(() => { void nativeSelected; void refresh().catch((cause) => error = String(cause)); });
  onMount(() => {
    let disposed = false;
    void invoke<Component[]>("get_native_model_catalog", { serviceId: "carey" })
      .then((entries) => { if (!disposed) catalog = entries; })
      .catch((cause) => { if (!disposed) error = String(cause); });
    const listener = listen<{ serviceId: string; encoding: string }>("native-model-selected", (event) => {
      if (!disposed && event.payload.serviceId === "carey") activeEncoding = event.payload.encoding;
    });
    return () => { disposed = true; void listener.then((stop) => stop()); };
  });
  async function useTier(tier: string) {
    busy = true; error = null; message = null;
    try {
      const selection = await invoke<{ encoding: string }>("select_native_model", { serviceId: "carey", encoding: tier });
      activeEncoding = selection.encoding;
      message = `Carey will need the 2B base at ${tier} next time it starts.`;
    } catch (cause) { error = String(cause); }
    finally { busy = false; }
  }
</script>

{#snippet actions(entry: Component, tier: string | null)}
  {@const model = models.find((item) => item.id === entry.id)}
  {@const download = progress.get(entry.id)}
  {#if model?.status === "downloading"}
    <DownloadProgress value={download?.progress ?? 0} label={`${entry.label} download`} />
  {:else if model?.status === "downloaded"}
    <div class="actions">
      {#if tier && nativeSelected && activeEncoding !== tier}<button class="download" disabled={busy || downloading || removingId !== null || status("carey-native::shared") !== "downloaded"} title={`Require the 2B base at ${tier} next time Carey starts`} onclick={() => useTier(tier)}>use</button>{/if}
      <button disabled={busy || downloading || removingId !== null || (tier !== null && nativeSelected && activeEncoding === tier)} title={tier !== null && nativeSelected && activeEncoding === tier ? "Carey needs this model to start" : undefined} onclick={() => onRemove(entry.id)}>{removingId === entry.id ? "removing..." : "remove"}</button>
    </div>
  {:else}<button class="download" disabled={busy || downloading || removingId !== null} onclick={() => onDownload(entry.id)}>{model?.status === "failed" ? "Retry" : "Download"}</button>{/if}
{/snippet}

{#if shared}
  {@const download = progress.get(shared.id)}
  <section aria-label="shared components">
    <div class="group-label">shared components</div>
    <p>Every model needs these. Download them once.</p>
    <div class="model-row" class:downloaded={status(shared.id) === "downloaded"}>
      <div class="model-info">
        <span class="model-name">{shared.label}</span>
        <span class="model-size">{shared.description} · {size(bytesOf(shared))}{status(shared.id) === "downloaded" ? " on disk" : " download"}</span>
        {#if status(shared.id) === "downloading"}<span class="model-size" role="status">{download?.message ?? "Queued for download"}</span>{/if}
        {#if download?.error}<span class="error" role="alert">{download.error}</span>{/if}
      </div>
      {@render actions(shared, null)}
    </div>
  </section>
{/if}

{#each groups as group (group.label)}
  <section aria-label={group.label}>
    <div class="group-label">{group.label}</div>
    <p>{group.hint}
      {#if group.variants[0] === "base" && activeEncoding}Carey is set to the 2B base at {activeEncoding}. Stop Carey before changing it.{/if}</p>
    {#each group.variants as variant (variant)}
      {@const tier = shown(variant)}
      {@const entry = catalog.find((item) => item.id === `carey-native::${variant}::${tier}`)}
      {#if entry}
        {@const download = progress.get(entry.id)}
        {@const others = onDisk(variant).filter((other) => other !== tier)}
        <div class="model-row" class:downloaded={status(entry.id) === "downloaded"}>
          <div class="model-info">
            <span class="model-name">{entry.label.replace(` ${tier}`, "")}
              <select aria-label={`${entry.label.replace(` ${tier}`, "")} precision`} value={tier} disabled={busy} onchange={(event) => picked[variant] = event.currentTarget.value}>
                {#each precisions(variant) as option (option)}<option value={option}>{option}{recommended.components.includes(`carey-native::${variant}::${option}`) ? " (recommended)" : ""}</option>{/each}
              </select>
              {#if variant === "base" && activeEncoding === tier}<span class="badge active">active</span>{/if}
              {#if recommended.components.includes(entry.id)}<span class="badge">recommended</span>{/if}
            </span>
            <span class="model-size">{size(bytesOf(entry))}{status(entry.id) === "downloaded" ? " on disk" : " download"}{others.length ? ` · also on disk: ${others.join(", ")}` : ""}</span>
            {#if status(entry.id) === "downloading"}<span class="model-size" role="status">{download?.message ?? "Queued for download"}</span>{/if}
            {#if download?.error}<span class="error" role="alert">{download.error}</span>{/if}
          </div>
          {@render actions(entry, variant === "base" ? tier : null)}
        </div>
      {/if}
    {/each}
  </section>
{/each}
{#if !recommended.tier}<p>Recommendations appear once the C++ runtime is installed and has checked this GPU.</p>{/if}
{#if !nativeSelected}<p>Use migrate runtime on the Carey service row to switch to C++ after downloading.</p>{/if}
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
  .model-name { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; font-size: 12px; color: var(--text-primary); }
  .model-name select { font-size: 11px; padding: 1px 4px; }
  .model-size { font-size: 10px; color: var(--text-muted); overflow-wrap: anywhere; }
  .actions { display: flex; gap: 6px; }
  button { flex-shrink: 0; font-size: 11px; }
  button.download { color: var(--accent); border-color: var(--accent); }
  .badge { display: inline-block; padding: 1px 4px; border: 1px solid var(--border); border-radius: 3px; font-size: 9px; color: var(--text-muted); }
  .badge.active { color: var(--green); border-color: var(--green); }
  .error { color: var(--red); font-size: 11px; overflow-wrap: anywhere; }
</style>
