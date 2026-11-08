<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";
  import { recommendedModelIds } from "./runtimeSetup";
  let { serviceId, ready, busy, onDownload }: {
    serviceId: string;
    ready: boolean;
    busy: boolean;
    onDownload: () => void;
  } = $props();
  interface Model { id: string; status: string; }
  let models = $state<Model[]>([]);
  let plan = $state<string[] | null>(null);
  let error = $state<string | null>(null);
  let revision = 0;
  const complete = $derived(plan !== null && (serviceId === "yuey"
    ? models.some((model) => model.id === "yuey::shared" && model.status === "downloaded")
      && models.some((model) => ["yuey::BF16", "yuey::Q8_0", "yuey::Q4_K_M"].includes(model.id) && model.status === "downloaded")
    : plan.every((id) => models.some((model) => model.id === id && model.status === "downloaded"))));
  onMount(() => {
    let disposed = false;
    const listener = listen<Model[]>("models-updated", (event) => {
      revision++;
      if (!disposed) models = event.payload;
    });
    return () => { disposed = true; void listener.then((unlisten) => unlisten()); };
  });
  $effect(() => {
    const id = serviceId;
    if (!ready) { plan = null; return; }
    let disposed = false;
    const before = revision;
    void Promise.all([recommendedModelIds(id), invoke<Model[]>("get_models")]).then(([ids, current]) => {
      if (disposed) return;
      plan = ids;
      if (revision === before) models = current;
      error = null;
    }).catch((cause) => { if (!disposed) error = String(cause); });
    return () => { disposed = true; };
  });
</script>

{#if !ready || error || (plan !== null && !complete)}
<div class="model-actions" aria-label="Model downloads">
  <button type="button" disabled={!ready || busy} title={!ready ? "Install the runtime first so it can recommend models for your hardware." : undefined} onclick={onDownload}>{busy ? "queuing models…" : "download recommended models"}</button>
  {#if error}<span class="error" role="alert">{error}</span>{/if}
</div>
{/if}

<style>
  .model-actions { display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 16px; border-bottom: 1px solid var(--border); background: #1a2028; }
  button { padding: 2px 8px; border: 1px solid var(--border); border-radius: 3px; background: transparent; color: var(--text-primary); font-size: 11px; }
  button:disabled { color: var(--text-muted); }
  .error { font-size: 11px; color: var(--red); overflow-wrap: anywhere; }
</style>
