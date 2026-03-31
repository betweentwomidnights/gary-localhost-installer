<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";

  interface ModelEntry {
    id: string;
    display_name: string;
    service: string;
    size_category: string | null;
    group: string | null;
    epoch: number | null;
    status: "available" | "downloading" | "downloaded" | "failed";
  }

  interface DownloadProgress {
    model_id: string;
    progress: number;
    status: string;
    message: string;
    error: string | null;
  }

  let { serviceId, onBack }: {
    serviceId: string;
    onBack: () => void;
  } = $props();

  let models: ModelEntry[] = $state([]);
  let progress: Map<string, DownloadProgress> = $state(new Map());
  let filterSize: string = $state("all");

  // Jerry finetune state
  let finetuneRepo: string = $state("thepatch/jerry_grunge");
  let fetchingCheckpoints: boolean = $state(false);
  let fetchError: string | null = $state(null);

  const isJerry = $derived(serviceId === "stable-audio");
  const isCarey = $derived(serviceId === "carey");

  async function loadModels() {
    try {
      models = await invoke<ModelEntry[]>("get_models");
    } catch (e) {
      console.error("Failed to load models:", e);
    }
  }

  async function startDownload(modelId: string) {
    try {
      await invoke("download_model", { modelId, serviceId });
      await loadModels();
    } catch (e) {
      console.error("Download failed:", e);
    }
  }

  async function fetchCheckpoints() {
    const repo = finetuneRepo.trim();
    if (!repo) return;

    fetchingCheckpoints = true;
    fetchError = null;
    try {
      await invoke<string[]>("fetch_jerry_checkpoints", { repo });
      await loadModels();
    } catch (e: any) {
      fetchError = typeof e === "string" ? e : e?.message || "Failed to fetch checkpoints";
      console.error("Fetch checkpoints failed:", e);
    } finally {
      fetchingCheckpoints = false;
    }
  }

  onMount(() => {
    loadModels();

    const unlistenModels = listen<ModelEntry[]>("models-updated", (event) => {
      models = event.payload;
    });

    const unlistenProgress = listen<DownloadProgress[]>("download-progress", (event) => {
      const newMap = new Map<string, DownloadProgress>();
      for (const p of event.payload) {
        newMap.set(p.model_id, p);
      }
      progress = newMap;
    });

    const pollTimer = setInterval(async () => {
      try {
        const prog = await invoke<DownloadProgress[]>("get_download_progress");
        const newMap = new Map<string, DownloadProgress>();
        for (const p of prog) {
          newMap.set(p.model_id, p);
        }
        progress = newMap;
        await loadModels();
      } catch (_) {}
    }, 3000);

    return () => {
      clearInterval(pollTimer);
      unlistenModels.then((fn) => fn());
      unlistenProgress.then((fn) => fn());
    };
  });

  // Filter models belonging to this service
  let serviceModels = $derived(
    models.filter((m) => m.service === serviceId)
  );

  // For Jerry: separate base models from finetune checkpoints
  let baseModels = $derived(
    serviceModels.filter((m) => m.size_category === "base")
  );
  let finetuneModels = $derived(
    serviceModels.filter((m) => m.size_category === "finetune")
  );

  let filteredModels = $derived(
    filterSize === "all"
      ? serviceModels
      : serviceModels.filter((m) => m.size_category === filterSize)
  );

  let downloadedCount = $derived(serviceModels.filter((m) => m.status === "downloaded").length);
  let activeDownloads = $derived(serviceModels.filter((m) => m.status === "downloading").length);

  function getProgress(modelId: string): DownloadProgress | undefined {
    return progress.get(modelId);
  }

  // Derive service display name
  let serviceLabel = $derived(
    serviceId === "gary" ? "gary (musicgen)" :
    serviceId === "stable-audio" ? "jerry (stable audio)" :
    serviceId === "carey" ? "carey (ace-step)" :
    serviceId === "foundation" ? "foundation-1" :
    serviceId
  );

  const statusLabels: Record<string, string> = {
    available: "Download",
    downloading: "Downloading...",
    downloaded: "Downloaded",
    failed: "Retry",
  };

  const sizeOrder = ["small", "medium", "large"];

  // Check which sizes exist for this service
  let availableSizes = $derived(
    [...new Set(serviceModels.map(m => m.size_category).filter(Boolean))] as string[]
  );
</script>

<div class="model-panel">
  <div class="panel-header">
    <div class="header-top">
      <button class="back-btn" onclick={onBack}>&larr; Back to logs</button>
    </div>
    <div class="header-info">
      <h2>{serviceLabel} models</h2>
      <span class="count">{downloadedCount}/{serviceModels.length} downloaded</span>
      {#if activeDownloads > 0}
        <span class="active-badge">{activeDownloads} downloading</span>
      {/if}
    </div>
    {#if !isJerry && !isCarey && availableSizes.length > 1}
      <div class="filters">
        <button class:active={filterSize === "all"} onclick={() => filterSize = "all"}>All</button>
        {#each sizeOrder as size}
          {#if availableSizes.includes(size)}
            <button class:active={filterSize === size} onclick={() => filterSize = size}>{size}</button>
          {/if}
        {/each}
      </div>
    {/if}
  </div>

  <div class="model-list">
    {#if isJerry}
      <!-- Jerry: Base model section -->
      <div class="size-group">
        <div class="size-label">base model</div>
        {#each baseModels as model}
          {@const prog = getProgress(model.id)}
          <div class="model-row" class:downloaded={model.status === "downloaded"} class:downloading={model.status === "downloading"}>
            <div class="model-info">
              <span class="model-name">{model.display_name}</span>
              <span class="model-path">{model.id}</span>
            </div>
            {#if model.status === "downloading" && prog}
              <div class="download-progress">
                <div class="progress-bar">
                  <div class="progress-fill" style="width: {Math.round(prog.progress * 100)}%"></div>
                </div>
                <span class="progress-pct">{Math.round(prog.progress * 100)}%</span>
              </div>
            {:else}
              <button
                class="dl-btn"
                class:downloaded={model.status === "downloaded"}
                class:failed={model.status === "failed"}
                disabled={model.status === "downloading" || model.status === "downloaded"}
                onclick={(e) => { e.stopPropagation(); startDownload(model.id); }}
              >
                {statusLabels[model.status] || "Download"}
              </button>
            {/if}
          </div>
        {/each}
      </div>

      <!-- Jerry: Finetune section -->
      <div class="size-group">
        <div class="size-label">finetune checkpoints</div>
        <div class="finetune-input">
          <input
            type="text"
            bind:value={finetuneRepo}
            placeholder="hugging face repo (e.g. thepatch/jerry_grunge)"
            onkeydown={(e) => e.key === "Enter" && fetchCheckpoints()}
          />
          <button
            class="fetch-btn"
            onclick={fetchCheckpoints}
            disabled={fetchingCheckpoints || !finetuneRepo.trim()}
          >
            {fetchingCheckpoints ? "Fetching..." : "Fetch Checkpoints"}
          </button>
        </div>
        {#if fetchError}
          <div class="fetch-error">{fetchError}</div>
        {/if}
        {#if finetuneModels.length > 0}
          {#each finetuneModels as model}
            {@const prog = getProgress(model.id)}
            <div class="model-row" class:downloaded={model.status === "downloaded"} class:downloading={model.status === "downloading"}>
              <div class="model-info">
                <span class="model-name">{model.display_name}</span>
                <span class="model-path">{model.group || ""}</span>
              </div>
              {#if model.status === "downloading" && prog}
                <div class="download-progress">
                  <div class="progress-bar">
                    <div class="progress-fill" style="width: {Math.round(prog.progress * 100)}%"></div>
                  </div>
                  <span class="progress-pct">{Math.round(prog.progress * 100)}%</span>
                </div>
              {:else}
                <button
                  class="dl-btn"
                  class:downloaded={model.status === "downloaded"}
                  class:failed={model.status === "failed"}
                  disabled={model.status === "downloading" || model.status === "downloaded"}
                  onclick={(e) => { e.stopPropagation(); startDownload(model.id); }}
                >
                  {statusLabels[model.status] || "Download"}
                </button>
              {/if}
            </div>
          {/each}
        {:else if !fetchingCheckpoints}
          <div class="finetune-hint">
            Enter a HuggingFace repo ID and click "Fetch Checkpoints" to see available .ckpt files.
            jerry must be running to fetch checkpoints.
          </div>
        {/if}
      </div>

    {:else if isCarey}
      <!-- Carey: base model components -->
      <div class="size-group">
        <div class="size-label">base model components</div>
        <div class="carey-hint">
          download all three components before starting carey.
          Models are stored in services/carey/checkpoints/.
        </div>
        {#each serviceModels as model}
          {@const prog = getProgress(model.id)}
          <div class="model-row" class:downloaded={model.status === "downloaded"} class:downloading={model.status === "downloading"}>
            <div class="model-info">
              <span class="model-name">{model.display_name}</span>
              <span class="model-path">{model.id.replace("carey::", "")}</span>
            </div>
            {#if model.status === "downloading" && prog}
              <div class="download-progress">
                <div class="progress-bar">
                  <div class="progress-fill" style="width: {Math.round(prog.progress * 100)}%"></div>
                </div>
                <span class="progress-pct">{Math.round(prog.progress * 100)}%</span>
              </div>
            {:else}
              <button
                class="dl-btn"
                class:downloaded={model.status === "downloaded"}
                class:failed={model.status === "failed"}
                disabled={model.status === "downloading" || model.status === "downloaded"}
                onclick={(e) => { e.stopPropagation(); startDownload(model.id); }}
              >
                {statusLabels[model.status] || "Download"}
              </button>
            {/if}
          </div>
        {/each}
      </div>

    {:else}
      <!-- Gary / other services: grouped by size -->
      {#if serviceModels.length === 0}
        <div class="empty">
          <p>No models configured for this service yet.</p>
        </div>
      {:else}
        {#each sizeOrder as size}
          {@const sizeModels = filteredModels.filter(m => m.size_category === size)}
          {#if sizeModels.length > 0}
            <div class="size-group">
              <div class="size-label">{size}</div>
              {#each sizeModels as model}
                {@const prog = getProgress(model.id)}
                <div class="model-row" class:downloaded={model.status === "downloaded"} class:downloading={model.status === "downloading"}>
                  <div class="model-info">
                    <span class="model-name">{model.display_name}</span>
                    <span class="model-path">{model.id}</span>
                  </div>

                  {#if model.status === "downloading" && prog}
                    <div class="download-progress">
                      <div class="progress-bar">
                        <div class="progress-fill" style="width: {Math.round(prog.progress * 100)}%"></div>
                      </div>
                      <span class="progress-pct">{Math.round(prog.progress * 100)}%</span>
                    </div>
                  {:else}
                    <button
                      class="dl-btn"
                      class:downloaded={model.status === "downloaded"}
                      class:failed={model.status === "failed"}
                      disabled={model.status === "downloading" || model.status === "downloaded"}
                      onclick={(e) => { e.stopPropagation(); startDownload(model.id); }}
                    >
                      {statusLabels[model.status] || "Download"}
                    </button>
                  {/if}
                </div>
              {/each}
            </div>
          {/if}
        {/each}
        <!-- Models without size category -->
        {@const uncategorized = filteredModels.filter(m => !m.size_category || !sizeOrder.includes(m.size_category))}
        {#if uncategorized.length > 0}
          <div class="size-group">
            <div class="size-label">models</div>
            {#each uncategorized as model}
              {@const prog = getProgress(model.id)}
              <div class="model-row" class:downloaded={model.status === "downloaded"} class:downloading={model.status === "downloading"}>
                <div class="model-info">
                  <span class="model-name">{model.display_name}</span>
                  <span class="model-path">{model.id}</span>
                </div>
                {#if model.status === "downloading" && prog}
                  <div class="download-progress">
                    <div class="progress-bar">
                      <div class="progress-fill" style="width: {Math.round(prog.progress * 100)}%"></div>
                    </div>
                    <span class="progress-pct">{Math.round(prog.progress * 100)}%</span>
                  </div>
                {:else}
                  <button
                    class="dl-btn"
                    class:downloaded={model.status === "downloaded"}
                    class:failed={model.status === "failed"}
                    disabled={model.status === "downloading" || model.status === "downloaded"}
                    onclick={(e) => { e.stopPropagation(); startDownload(model.id); }}
                  >
                    {statusLabels[model.status] || "Download"}
                  </button>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      {/if}
    {/if}
  </div>
</div>

<style>
  .model-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .panel-header {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-secondary);
  }

  .header-top {
    margin-bottom: 6px;
  }

  .back-btn {
    font-size: 11px;
    padding: 2px 8px;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
  }

  .back-btn:hover {
    color: var(--text-primary);
  }

  .header-info {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 6px;
  }

  .header-info h2 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .count {
    font-size: 11px;
    color: var(--text-secondary);
    font-family: var(--font-mono);
  }

  .active-badge {
    font-size: 10px;
    padding: 1px 6px;
    background: var(--accent);
    color: white;
    border-radius: 3px;
    font-family: var(--font-mono);
  }

  .filters {
    display: flex;
    gap: 4px;
  }

  .filters button {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 3px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
  }

  .filters button.active {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }

  .model-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }

  .empty {
    padding: 40px 20px;
    text-align: center;
    color: var(--text-muted);
    font-size: 12px;
  }

  .size-group {
    margin-bottom: 4px;
  }

  .size-label {
    padding: 4px 16px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
  }

  .model-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 16px;
    gap: 8px;
    transition: background 0.1s;
  }

  .model-row:hover {
    background: var(--bg-hover);
  }

  .model-row.downloaded {
    opacity: 0.7;
  }

  .model-info {
    display: flex;
    flex-direction: column;
    gap: 1px;
    min-width: 0;
    flex: 1;
  }

  .model-name {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .model-path {
    font-size: 10px;
    color: var(--text-muted);
    font-family: var(--font-mono);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .dl-btn {
    flex-shrink: 0;
    font-size: 10px;
    padding: 3px 10px;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: var(--bg-panel);
    color: var(--text-primary);
    cursor: pointer;
  }

  .dl-btn:hover:not(:disabled) {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }

  .dl-btn.downloaded {
    color: var(--green);
    border-color: var(--green);
    opacity: 0.6;
    cursor: default;
  }

  .dl-btn.failed {
    color: var(--red);
    border-color: var(--red);
  }

  .dl-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .download-progress {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
    width: 140px;
  }

  .progress-bar {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  .progress-pct {
    font-size: 10px;
    color: var(--text-secondary);
    font-family: var(--font-mono);
    width: 28px;
    text-align: right;
  }

  /* Jerry finetune input styles */
  .finetune-input {
    display: flex;
    gap: 6px;
    padding: 6px 16px;
    align-items: center;
  }

  .finetune-input input {
    flex: 1;
    font-size: 11px;
    padding: 5px 8px;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: var(--bg-panel);
    color: var(--text-primary);
    font-family: var(--font-mono);
    outline: none;
  }

  .finetune-input input:focus {
    border-color: var(--accent);
  }

  .finetune-input input::placeholder {
    color: var(--text-muted);
  }

  .fetch-btn {
    flex-shrink: 0;
    font-size: 10px;
    padding: 5px 12px;
    border: 1px solid var(--accent);
    border-radius: 3px;
    background: transparent;
    color: var(--accent);
    cursor: pointer;
    white-space: nowrap;
  }

  .fetch-btn:hover:not(:disabled) {
    background: var(--accent);
    color: white;
  }

  .fetch-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .fetch-error {
    padding: 4px 16px;
    font-size: 11px;
    color: var(--red);
  }

  .finetune-hint {
    padding: 12px 16px;
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.5;
  }

  .carey-hint {
    padding: 4px 16px 8px;
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.4;
  }

</style>
