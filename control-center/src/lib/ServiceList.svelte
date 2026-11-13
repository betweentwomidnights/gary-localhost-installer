<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";
  import ServiceRow from "./ServiceRow.svelte";
  import { runtimeGroupLabel } from "./runtimeGroups";

  interface BuildStatus {
    building: boolean;
    current_step: number;
    total_steps: number;
    step_label: string;
    log: string;
    error: string | null;
  }

  interface ServiceInfo {
    id: string;
    display_name: string;
    port: number;
    status: "stopped" | "starting" | "running" | "unhealthy" | "failed";
    pid: number | null;
    error: string | null;
    env_exists: boolean;
    build_status: BuildStatus | null;
    runtime: "python" | "native";
    native_backend: string | null;
    native_runtime_installed: boolean;
    native_update_available: boolean;
    native_fallback_reason: string | null;
    start_blocker: string | null;
    start_activation_encoding?: string | null;
    migration_complete: boolean;
    has_migration: boolean;
    runtime_group: string | null;
    python_retired: boolean;
  }

  let { services, selectedServiceId, selectedRuntimeGroup, hfTokenConfigured, onSelect, onSelectGroup, onShowModels, onManageCareyLoras, onTrainCareyAce, onManageSa3Loras, onTrainSa3Lora, onMigrateRuntime }: {
    services: ServiceInfo[];
    selectedServiceId: string | null;
    selectedRuntimeGroup: string | null;
    onSelectGroup: (group: string) => void;
    hfTokenConfigured: boolean;
    onSelect: (id: string) => void;
    onShowModels: (id: string) => void;
    onManageCareyLoras: () => void;
    onTrainCareyAce: () => void;
    onManageSa3Loras: () => void;
    onTrainSa3Lora: () => void;
    onMigrateRuntime: (id: string) => void;
  } = $props();

  // Services never set up (no Python to migrate), and C++ runtimes that are
  // missing or behind; services sharing a runtime update it once.
  const toSetUp = $derived(services.filter((service) => service.has_migration && service.runtime === "python" && service.python_retired && !service.native_runtime_installed));
  const toUpdate = $derived(services.filter((service) => (service.runtime === "native" || service.python_retired) && (service.native_update_available || (service.runtime === "native" && !service.env_exists))));
  let runningAll = $state(false);
  let choosing = $state(false);
  let loadingChoices = $state(false);
  let choices = $state<{ id: string; offeredBackends: string[] }[]>([]);
  let backend = $state("cuda");
  let allProgress = $state("");
  let allErrors = $state<string[]>([]);
  const activeBuild = $derived(services.find((service) => service.build_status?.building));
  interface Model { id: string; status: string; }
  let models = $state<Model[]>([]);
  onMount(() => {
    let disposed = false;
    let revision = 0;
    const listener = listen<Model[]>("models-updated", (event) => {
      revision++;
      if (!disposed) models = event.payload;
    });
    void listener.then(async () => {
      const before = revision;
      const current = await invoke<Model[]>("get_models");
      if (!disposed && revision === before) models = current;
    }).catch(console.error);
    return () => { disposed = true; void listener.then((unlisten) => unlisten()); };
  });
  function trainingBlocker(service: ServiceInfo): string | null {
    const native = service.runtime === "native" || service.python_retired;
    if (!(native ? service.native_runtime_installed : service.env_exists)) return "Install the runtime first.";
    if (service.build_status?.building) return "Wait for runtime checks to finish.";
    const ready = (id: string) => models.some((model) => model.id === id && model.status === "downloaded");
    if (native && service.id === "sa3" && !(ready("sa3-native::text") && ready("sa3-native::medium-decoder") && ["F16", "Q4_K_M"].some((tier) => ready(`sa3-native::medium-base-${tier}`)))) return "Download SA3's shared components and a medium training base from Models.";
    if (native && service.id === "carey" && !(ready("carey-native::shared") && models.some((model) => /^carey-native::(base|xl-base)::/.test(model.id) && model.status === "downloaded"))) return "Download Carey's shared components and a 2B or XL base from Models.";
    if (!native && service.id === "sa3" && (!hfTokenConfigured || !ready("stabilityai/stable-audio-3-medium-base"))) return "Set up Hugging Face access and download SA3's medium training base.";
    return null;
  }
  const allLabel = $derived(
    runningAll ? "working…"
    : toSetUp.length ? "set up all runtimes"
    : toUpdate.length ? "update all runtimes"
    : "runtimes up to date"
  );

  async function reviewSetup() {
    choosing = true;
    loadingChoices = true;
    choices = [];
    allErrors = [];
    try {
      const seen = new Set<string>();
      const pending = [...toSetUp, ...toUpdate].filter((service) => {
        const key = service.runtime_group ?? service.id;
        if (seen.has(key) || service.native_runtime_installed) return false;
        seen.add(key);
        return true;
      });
      choices = await Promise.all(pending.map(async (service) => {
        const info = await invoke<{ offeredBackends: string[] }>("get_native_runtime_info", { serviceId: service.id });
        return { id: service.id, offeredBackends: info.offeredBackends };
      }));
      if (!choices.length) {
        await setUpAll();
      } else {
        const gpus = await invoke<{ vendor: string }[]>("get_native_setup_gpus");
        if (!gpus.some((gpu) => gpu.vendor === "nvidia") || !choices.some((choice) => choice.offeredBackends.includes("cuda"))) {
          backend = "vulkan";
          await setUpAll();
        } else {
          backend = "cuda";
        }
      }
    } catch (e) { allErrors = [String(e)]; }
    finally { loadingChoices = false; }
  }

  async function setUpAll() {
    runningAll = true;
    allErrors = [];
    allProgress = "Preparing runtimes…";
    try {
      if (choices.length) await invoke("save_app_settings", { settings: { nativeBackends: Object.fromEntries(choices.map((choice) => [choice.id, choice.offeredBackends.includes(backend) ? backend : choice.offeredBackends.includes("vulkan") ? "vulkan" : "auto"])) } });
      choosing = false;
      await invoke("update_native_runtimes");
      allProgress = "";
    } catch (e) {
      allErrors = String(e).split("\n");
      allProgress = "Runtime setup needs attention. You can retry the remaining runtimes.";
    } finally {
      runningAll = false;
    }
  }

  // Jerry's and SA3's Python models come from gated Hugging Face repos, so
  // their models button waits for a token. Their C++ models are public.
  const pythonModelsNeedToken = (service: ServiceInfo) =>
    ["stable-audio", "sa3"].includes(service.id) && service.runtime === "python" && !service.python_retired;

  // Services that have downloadable models
  const servicesWithModels = new Set(["gary", "melodyflow", "stable-audio", "sa3", "carey", "foundation", "yuey"]);

  // Services sharing a native runtime are listed together under it, where
  // the first of them would have been.
  type Entry = { service: ServiceInfo } | { group: string; members: ServiceInfo[] };
  const entries = $derived.by(() => {
    const out: Entry[] = [];
    const seen = new Set<string>();
    for (const service of services) {
      const group = service.runtime_group;
      if (!group) { out.push({ service }); continue; }
      if (seen.has(group)) continue;
      seen.add(group);
      out.push({ group, members: services.filter((member) => member.runtime_group === group) });
    }
    return out;
  });
  const live = (service: ServiceInfo) => ["running", "starting", "unhealthy"].includes(service.status);
  function groupSummary(members: ServiceInfo[]) {
    const native = members.find((member) => member.native_runtime_installed);
    const parts = [`${members.filter(live).length}/${members.length} running`];
    if (native?.native_backend) parts.push(native.native_backend);
    if (members.some((member) => member.native_update_available)) parts.push("update available");
    if (members.some((member) => member.build_status?.building)) parts.push("busy");
    return parts.join(" · ");
  }
</script>

{#snippet row(service: ServiceInfo, grouped: boolean)}
  <ServiceRow
    {service}
    {grouped}
    selected={selectedServiceId === service.id}
    onSelect={() => onSelect(service.id)}
    hasModels={servicesWithModels.has(service.id) && (!pythonModelsNeedToken(service) || hfTokenConfigured)}
    onShowModels={() => onShowModels(service.id)}
    hasCareyLoras={service.id === "carey"}
    onManageCareyLoras={onManageCareyLoras}
    hasCareyAceTraining={service.id === "carey"}
    onTrainCareyAce={onTrainCareyAce}
    hasSa3Loras={service.id === "sa3"}
    onManageSa3Loras={onManageSa3Loras}
    hasSa3LoraTraining={service.id === "sa3"}
    trainingBlocker={trainingBlocker(service)}
    onTrainSa3Lora={onTrainSa3Lora}
    hasMigration={service.has_migration && !service.migration_complete}
    onMigrateRuntime={() => onMigrateRuntime(service.id)}
  />
{/snippet}

<div class="service-list">
  <div class="list-header">
    <span class="label">services</span>
    <button onclick={reviewSetup} disabled={runningAll || choosing || !!activeBuild || (!toSetUp.length && !toUpdate.length)}>{allLabel}</button>
  </div>
  {#if choosing}
    <section class="setup-options" aria-label="Runtime setup options">
      <strong>Set up runtimes</strong>
      <p>Installs each shared runtime once. Models are separate: choose them per service afterward.</p>
      <p>Vulkan uses a smaller download. CUDA needs a larger shared NVIDIA runtime plus a backend for each engine and may run faster. You can switch later in runtime settings.</p>
      {#if loadingChoices}<p role="status">Checking available backends…</p>{/if}
      {#if !loadingChoices}
        <div class="setup-choice" role="group" aria-label="Backend for all new runtimes">
          <label><input type="radio" bind:group={backend} value="cuda" disabled={runningAll} /> CUDA</label>
          <label><input type="radio" bind:group={backend} value="vulkan" disabled={runningAll} /> Vulkan</label>
        </div>
      {/if}
      <button onclick={() => choosing = false} disabled={runningAll || loadingChoices}>cancel</button>
      <button onclick={setUpAll} disabled={runningAll || loadingChoices || !!allErrors.length}>install runtimes</button>
    </section>
  {/if}
  {#if allProgress}<div class="all-status" role="status">{#if runningAll}<span class="spinner" aria-hidden="true"></span>{/if}{runningAll && activeBuild ? `${activeBuild.display_name}: ${activeBuild.build_status?.step_label}` : allProgress}</div>{/if}
  {#each allErrors as message}<div class="all-error" role="alert">{message}</div>{/each}
  {#each entries as entry ("group" in entry ? `group:${entry.group}` : entry.service.id)}
    {#if "group" in entry}
      <div class="runtime-group">
        <button class="group-header" class:selected={selectedRuntimeGroup === entry.group} onclick={() => onSelectGroup(entry.group)}>
          <span class="group-name">{runtimeGroupLabel(entry.group)}</span>
          <span class="group-meta">{groupSummary(entry.members)}</span>
        </button>
        <div class="group-members">
          {#each entry.members as service (service.id)}{@render row(service, true)}{/each}
        </div>
      </div>
    {:else}
      {@render row(entry.service, false)}
    {/if}
  {/each}
  {#if services.length === 0}
    <div class="empty">loading services...</div>
  {/if}
</div>

<style>
  .service-list {
    padding: 8px;
  }
  .list-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    margin-bottom: 4px;
  }
  .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary);
    font-weight: 600;
  }
  .all-status, .all-error {
    padding: 0 12px 6px;
    font-size: 11px;
    color: var(--text-secondary);
  }
  .all-error { color: var(--red); }
  .setup-options { margin: 8px 12px; padding: 12px; border: 1px solid var(--border); border-radius: 4px; font-size: 12px; }
  .setup-options p { color: var(--text-secondary); line-height: 1.5; }
  .setup-choice { display: flex; align-items: center; gap: 20px; margin: 12px 0; }
  .setup-choice label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
  /* Keep the exclusive choice and keyboard behavior of radios, with the
     same square accent check used by the application's other toggles. */
  .setup-choice input {
    appearance: none;
    display: grid;
    place-content: center;
    width: 13px;
    height: 13px;
    margin: 0;
    border: 1px solid var(--text-muted);
    border-radius: 2px;
    background: var(--bg-panel);
    cursor: pointer;
  }
  .setup-choice input:checked { background: var(--accent); border-color: var(--accent); }
  .setup-choice input::before {
    content: "";
    width: 3px;
    height: 7px;
    border: solid white;
    border-width: 0 2px 2px 0;
    transform: translateY(-1px) rotate(45deg);
    visibility: hidden;
  }
  .setup-choice input:checked::before { visibility: visible; }
  .setup-choice input:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
  .setup-choice input:disabled { opacity: 0.5; cursor: default; }
  .spinner { display: inline-block; width: 10px; height: 10px; margin-right: 6px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @media (prefers-reduced-motion: reduce) { .spinner { animation: none; } }
  .empty {
    padding: 20px;
    text-align: center;
    color: var(--text-muted);
  }
  .runtime-group { margin: 6px 0; }
  .group-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
    width: 100%;
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: transparent;
    color: var(--text-primary);
    text-align: left;
    cursor: pointer;
  }
  .group-header:hover { background: var(--bg-hover); }
  .group-header.selected { background: var(--bg-panel); border-color: var(--accent); }
  .group-name { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
  .group-meta { font-size: 10px; color: var(--text-secondary); font-family: var(--font-mono); white-space: nowrap; }
  .group-members { margin-left: 10px; padding-left: 4px; border-left: 1px solid var(--border); }
</style>
