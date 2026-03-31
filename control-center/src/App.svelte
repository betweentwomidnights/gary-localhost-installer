<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import ServiceList from "./lib/ServiceList.svelte";
  import LogViewer from "./lib/LogViewer.svelte";
  import ModelPanel from "./lib/ModelPanel.svelte";
  import TokenBanner from "./lib/TokenBanner.svelte";
  import MelodyflowFlashBanner from "./lib/MelodyflowFlashBanner.svelte";
  import CloseBehaviorModal from "./lib/CloseBehaviorModal.svelte";

  interface ServiceInfo {
    id: string;
    display_name: string;
    port: number;
    status: "stopped" | "starting" | "running" | "unhealthy" | "failed";
    pid: number | null;
    error: string | null;
    env_exists: boolean;
    health_endpoint: string | null;
  }

  interface AppSettings {
    melodyflowUseFlashAttn: boolean;
    closeActionOnX: "ask" | "tray" | "quit";
  }

  const showMelodyflowFlashBanner =
    import.meta.env.VITE_ENABLE_MELODYFLOW_FA2_TOGGLE !== "0";

  let services: ServiceInfo[] = $state([]);
  let selectedServiceId: string | null = $state(null);
  let logText: string = $state("");
  let pollTimer: number;
  let appSettings: AppSettings = $state({
    melodyflowUseFlashAttn: false,
    closeActionOnX: "ask",
  });
  let closeRequestModalOpen = $state(false);
  let rememberCloseChoice = $state(false);
  let resolvingCloseRequest = $state(false);

  // Right panel can show either logs or the model panel for a service
  let rightPanel: "logs" | "models" = $state("logs");
  let modelServiceId: string | null = $state(null);

  // HF token state — gates Jerry's Models button
  let hfTokenConfigured: boolean = $state(false);

  async function loadServices() {
    try {
      services = await invoke<ServiceInfo[]>("get_services");
    } catch (e) {
      console.error("Failed to load services:", e);
    }
  }

  async function fetchLog(serviceId: string) {
    try {
      logText = await invoke<string>("get_service_log", { serviceId });
    } catch (e) {
      logText = `Error reading log: ${e}`;
    }
  }

  function selectService(id: string) {
    selectedServiceId = id;
    rightPanel = "logs";
    fetchLog(id);
  }

  function showModels(serviceId: string) {
    selectedServiceId = serviceId;
    modelServiceId = serviceId;
    rightPanel = "models";
  }

  function backToLogs() {
    rightPanel = "logs";
    if (selectedServiceId) fetchLog(selectedServiceId);
  }

  async function checkToken() {
    try {
      const token = await invoke<string | null>("get_hf_token");
      hfTokenConfigured = !!token;
    } catch (_) {}
  }

  async function loadAppSettings() {
    try {
      appSettings = await invoke<AppSettings>("get_app_settings");
    } catch (e) {
      console.error("Failed to load app settings:", e);
    }
  }

  function onTokenChange(configured: boolean) {
    hfTokenConfigured = configured;
  }

  function onMelodyflowFlashSettingUpdated(enabled: boolean) {
    appSettings = { ...appSettings, melodyflowUseFlashAttn: enabled };
  }

  function onCloseRequestEvent() {
    closeRequestModalOpen = true;
    rememberCloseChoice = false;
  }

  function cancelCloseRequest() {
    if (resolvingCloseRequest) return;
    closeRequestModalOpen = false;
    rememberCloseChoice = false;
  }

  async function resolveCloseRequest(action: "tray" | "quit") {
    if (resolvingCloseRequest) return;
    resolvingCloseRequest = true;

    try {
      const updated = await invoke<AppSettings>("resolve_close_request", {
        action,
        rememberChoice: rememberCloseChoice,
      });
      appSettings = updated;
      closeRequestModalOpen = false;
      rememberCloseChoice = false;
    } catch (e) {
      console.error("Failed to resolve close request:", e);
    } finally {
      resolvingCloseRequest = false;
    }
  }

  onMount(() => {
    loadServices();
    checkToken();
    loadAppSettings();

    const unlisten = listen<ServiceInfo[]>("services-updated", (event) => {
      services = event.payload;
    });

    // When "Rebuild All" is running, the backend tells us which service to focus on
    const unlistenSelect = listen<string>("select-service", (event) => {
      selectService(event.payload);
    });

    const unlistenCloseRequest = listen("app-close-requested", () => {
      onCloseRequestEvent();
    });

    pollTimer = setInterval(() => {
      if (selectedServiceId && rightPanel === "logs") fetchLog(selectedServiceId);
    }, 2000);

    return () => {
      clearInterval(pollTimer);
      unlisten.then((fn) => fn());
      unlistenSelect.then((fn) => fn());
      unlistenCloseRequest.then((fn) => fn());
    };
  });

  let runningCount = $derived(services.filter((s) => s.status === "running").length);
  let totalCount = $derived(services.length);
  let selectedService = $derived(services.find((s) => s.id === selectedServiceId) ?? null);
</script>

<main>
  <header>
    <div class="header-left">
      <h1>gary4local</h1>
    </div>
    <div class="header-right">
      <span class="status-summary">
        {#if totalCount > 0}
          {runningCount}/{totalCount} running
        {/if}
      </span>
    </div>
  </header>
  <div class="panels">
    <div class="left-panel">
      <ServiceList
        {services}
        {selectedServiceId}
        {hfTokenConfigured}
        onSelect={selectService}
        onShowModels={showModels}
      />
    </div>
    <div class="divider"></div>
    <div class="right-panel">
      {#if rightPanel === "models" && modelServiceId}
        <ModelPanel serviceId={modelServiceId} onBack={backToLogs} />
      {:else}
        {#if selectedServiceId === "stable-audio"}
          <TokenBanner {onTokenChange} />
        {:else if selectedServiceId === "melodyflow" && showMelodyflowFlashBanner}
          <MelodyflowFlashBanner
            enabled={appSettings.melodyflowUseFlashAttn}
            serviceStatus={selectedService?.status ?? "stopped"}
            onUpdated={onMelodyflowFlashSettingUpdated}
          />
        {/if}
        <LogViewer
          serviceId={selectedServiceId}
          {logText}
        />
      {/if}
    </div>
  </div>
  <CloseBehaviorModal
    open={closeRequestModalOpen}
    rememberChoice={rememberCloseChoice}
    busy={resolvingCloseRequest}
    onRememberChange={(value) => rememberCloseChoice = value}
    onChoose={resolveCloseRequest}
    onCancel={cancelCloseRequest}
  />
</main>

<style>
  main {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-secondary);
    -webkit-app-region: drag;
  }
  header h1 {
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: var(--text-primary);
  }
  .header-right {
    -webkit-app-region: no-drag;
  }
  .status-summary {
    font-size: 11px;
    color: var(--text-secondary);
    font-family: var(--font-mono);
  }
  .panels {
    display: flex;
    flex: 1;
    overflow: hidden;
  }
  .left-panel {
    width: 420px;
    min-width: 320px;
    overflow-y: auto;
    border-right: 1px solid var(--border);
  }
  .divider {
    width: 1px;
    background: var(--border);
  }
  .right-panel {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
</style>
