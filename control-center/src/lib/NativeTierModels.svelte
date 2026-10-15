<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import DownloadProgress from "./DownloadProgress.svelte";
  import { onMount } from "svelte";

  interface Component { id: string; label: string; description: string; files: { bytes: number }[]; }
  interface Model { id: string; status: "available" | "downloading" | "downloaded" | "failed"; downloaded_bytes: number | null; }
  interface Progress { model_id: string; progress: number; status: string; message: string; error: string | null; }
  interface Profile {
    label: string;
    tiers: string[];
    tierComponents: Record<string, string[]>;
    tierOptions: Record<string, string[]>;
    optionsLabel: string | null;
  }
  let { serviceId, models, progress, nativeSelected, onDownload, onRemove, removingId }: {
    serviceId: string;
    models: Model[];
    progress: Map<string, Progress>;
    nativeSelected: boolean;
    onDownload: (id: string) => void;
    onRemove: (id: string) => void;
    removingId: string | null;
  } = $props();
  let profile = $state<Profile | null>(null);
  let catalog = $state<Component[]>([]);
  let activeEncoding = $state<string | null>(null);
  let busy = $state(false);
  let error = $state<string | null>(null);
  let message = $state<string | null>(null);
  const label = $derived(profile?.label ?? serviceId);
  // Each tier's own bundle comes first in its component list; what every
  // tier also needs (such as an encoder) is shared. A tier's options (such
  // as jerry's finetunes) run with it without being needed by it.
  const tierOf = $derived(new Map((profile?.tiers ?? []).map((tier) => [profile!.tierComponents[tier][0], tier])));
  const optionIds = $derived(new Set(Object.values(profile?.tierOptions ?? {}).flat()));
  const shared = $derived(catalog.filter((entry) => !tierOf.has(entry.id) && !optionIds.has(entry.id)));
  const tiers = $derived(catalog.filter((entry) => tierOf.has(entry.id)));
  const downloading = $derived(models.some((model) => model.status === "downloading"));
  const status = (id: string) => models.find((model) => model.id === id)?.status;
  const ready = (tier: string) => (profile?.tierComponents[tier] ?? []).every((id) => status(id) === "downloaded");
  // Options follow the tier the service runs (or will run) at: the active
  // one, else the first downloaded, else the default.
  const optionTier = $derived(activeEncoding
    ?? profile?.tiers.find((tier) => ready(tier))
    ?? profile?.tiers[0]
    ?? null);
  const options = $derived(optionTier
    ? (profile?.tierOptions[optionTier] ?? []).flatMap((id) => catalog.filter((entry) => entry.id === id))
    : []);
  function size(bytes: number) { return `${(bytes / 1024 ** 3).toFixed(2)} GB`; }

  $effect(() => {
    void nativeSelected;
    void invoke<{ encoding: string } | null>("get_native_runtime_selection", { serviceId })
      .then((selection) => activeEncoding = selection?.encoding ?? null)
      .catch((cause) => error = String(cause));
  });
  onMount(() => {
    let disposed = false;
    void Promise.all([
      invoke<Profile>("get_migration_profile", { serviceId }),
      invoke<Component[]>("get_native_model_catalog", { serviceId }),
    ]).then(([details, entries]) => { if (!disposed) { profile = details; catalog = entries; } })
      .catch((cause) => { if (!disposed) error = String(cause); });
    const listener = listen<{ serviceId: string; encoding: string }>("native-model-selected", (event) => {
      if (!disposed && event.payload.serviceId === serviceId) activeEncoding = event.payload.encoding;
    });
    return () => { disposed = true; void listener.then((stop) => stop()); };
  });
  async function useTier(tier: string) {
    busy = true; error = null; message = null;
    try {
      const selection = await invoke<{ encoding: string }>("select_native_model", { serviceId, encoding: tier });
      activeEncoding = selection.encoding;
      message = `${label} will use ${tier} next time it starts.`;
    } catch (cause) { error = String(cause); }
    finally { busy = false; }
  }
</script>

{#snippet row(entry: Component, tier: string | null)}
  {@const model = models.find((item) => item.id === entry.id)}
  {@const download = progress.get(entry.id)}
  <div class="model-row" class:downloaded={model?.status === "downloaded"}>
    <div class="model-info">
      <span class="model-name">{entry.label}
        {#if tier && activeEncoding === tier}<span class="badge active">active</span>
        {:else if tier && tier === profile?.tiers[0]}<span class="badge">default</span>{/if}
      </span>
      <span class="model-size">{entry.description} · {size(entry.files.reduce((sum, file) => sum + file.bytes, 0))}{model?.status === "downloaded" ? " on disk" : " download"}</span>
      {#if model?.status === "downloading"}<span class="model-size" role="status">{download?.message ?? "Queued for download"}</span>{/if}
      {#if download?.error}<span class="error" role="alert">{download.error}</span>{/if}
    </div>
    {#if model?.status === "downloading"}
      <DownloadProgress value={download?.progress ?? 0} label={`${entry.label} download`} />
    {:else if model?.status === "downloaded"}
      <div class="actions">
        {#if tier && nativeSelected && activeEncoding !== tier}<button class="download" disabled={busy || downloading || removingId !== null || !ready(tier)} title={ready(tier) ? `Use this tier next time ${label} starts` : "Download the shared components first"} onclick={() => useTier(tier)}>use</button>{/if}
        <button disabled={busy || downloading || removingId !== null} onclick={() => onRemove(entry.id)}>{removingId === entry.id ? "removing..." : "remove"}</button>
      </div>
    {:else}<button class="download" disabled={busy || downloading || removingId !== null} onclick={() => onDownload(entry.id)}>{model?.status === "failed" ? "Retry" : "Download"}</button>{/if}
  </div>
{/snippet}

{#if shared.length}
  <section aria-label="shared components">
    <div class="group-label">shared components</div>
    <p>Every tier needs these. Download them once.</p>
    {#each shared as entry (entry.id)}{@render row(entry, null)}{/each}
  </section>
{/if}
<section aria-label="generation tier">
  <div class="group-label">generation tier</div>
  <p>One tier is enough. {profile?.tiers[0] ?? "F16"} is the reference; smaller tiers use less memory and disk.
    {#if activeEncoding}{label} is set to {activeEncoding}. Stop {label} before changing its model.{/if}</p>
  {#each tiers as entry (entry.id)}{@render row(entry, tierOf.get(entry.id) ?? null)}{/each}
</section>
{#if options.length}
  <section aria-label={profile?.optionsLabel ?? "options"}>
    <div class="group-label">{profile?.optionsLabel ?? "options"}</div>
    <p>Each one downloaded at {optionTier} is offered in gary4juce alongside the base model.
      {#if optionTier && !ready(optionTier)}They run once the {optionTier} tier is downloaded.{/if}</p>
    {#each options as entry (entry.id)}{@render row(entry, null)}{/each}
  </section>
{/if}
{#if !nativeSelected}<p>Use migrate runtime on the {label} service row to switch to C++ after downloading.</p>{/if}
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
  .error { color: var(--red); font-size: 11px; overflow-wrap: anywhere; }
</style>
