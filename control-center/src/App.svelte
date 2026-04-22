<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import ServiceList from "./lib/ServiceList.svelte";
  import LogViewer from "./lib/LogViewer.svelte";
  import ModelPanel from "./lib/ModelPanel.svelte";
  import TokenBanner from "./lib/TokenBanner.svelte";
  import MelodyflowFlashBanner from "./lib/MelodyflowFlashBanner.svelte";
  import CareyXlBanner from "./lib/CareyXlBanner.svelte";
  import CareyLoraModal from "./lib/CareyLoraModal.svelte";
  import CloseBehaviorModal from "./lib/CloseBehaviorModal.svelte";
  import AppUpdateModal from "./lib/AppUpdateModal.svelte";

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
    health_endpoint: string | null;
    build_status: BuildStatus | null;
  }

  interface AppSettings {
    melodyflowUseFlashAttn: boolean;
    careyUseXlModels: boolean;
    closeActionOnX: "ask" | "tray" | "quit";
    autoCheckUpdates: boolean;
    skippedUpdateVersion: string | null;
    lastUpdateCheckEpochMs: number | null;
  }

  interface AppUpdateCheck {
    currentVersion: string;
    manifestUrl: string;
    checkedAtEpochMs: number;
    channel: string;
    latestVersion: string;
    updateAvailable: boolean;
    shouldPrompt: boolean;
    inAppInstallAvailable: boolean;
    downloadUrl: string | null;
    sha256: string | null;
    publishedAt: string | null;
    notes: string[];
  }

  const showMelodyflowFlashBanner =
    import.meta.env.VITE_ENABLE_MELODYFLOW_FA2_TOGGLE !== "0";
  const showAppUpdater = import.meta.env.VITE_ENABLE_APP_UPDATER !== "0";

  let services: ServiceInfo[] = $state([]);
  let selectedServiceId: string | null = $state(null);
  let logText: string = $state("");
  let pollTimer: number;
  let appSettings: AppSettings = $state({
    melodyflowUseFlashAttn: false,
    careyUseXlModels: false,
    closeActionOnX: "ask",
    autoCheckUpdates: true,
    skippedUpdateVersion: null,
    lastUpdateCheckEpochMs: null,
  });
  let closeRequestModalOpen = $state(false);
  let rememberCloseChoice = $state(false);
  let resolvingCloseRequest = $state(false);
  let updateCheckBusy = $state(false);
  let updateModalBusy = $state(false);
  let updateModalOpen = $state(false);
  let updateResult: AppUpdateCheck | null = $state(null);
  let updateCheckError: string | null = $state(null);
  let updateActionError: string | null = $state(null);
  let careyLoraModalOpen = $state(false);

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

  function showCareyLoras() {
    selectedServiceId = "carey";
    careyLoraModalOpen = true;
  }

  function closeCareyLoras() {
    careyLoraModalOpen = false;
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
    return appSettings;
  }

  function formatError(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }

  async function runUpdateCheck(options: {
    includeSkipped: boolean;
    openModalWhenCurrent: boolean;
    openModalOnError: boolean;
  }) {
    if (!showAppUpdater) return;

    updateCheckBusy = true;

    try {
      const result = await invoke<AppUpdateCheck>("check_for_app_update", {
        includeSkipped: options.includeSkipped,
      });

      if (options.openModalWhenCurrent || result.shouldPrompt) {
        updateResult = result;
        updateCheckError = null;
        updateActionError = null;
        updateModalOpen = true;
      } else if (result.updateAvailable) {
        updateResult = result;
        updateCheckError = null;
        updateActionError = null;
      }
    } catch (e) {
      const message = formatError(e);
      if (options.openModalOnError) {
        updateResult = null;
        updateCheckError = message;
        updateModalOpen = true;
      } else {
        console.warn("Update check failed:", message);
      }
    } finally {
      updateCheckBusy = false;
    }
  }

  async function checkForUpdatesManually() {
    await runUpdateCheck({
      includeSkipped: true,
      openModalWhenCurrent: true,
      openModalOnError: true,
    });
  }

  function closeUpdateModal() {
    updateModalOpen = false;
    updateCheckError = null;
    updateActionError = null;
  }

  async function setAutoCheckUpdates(enabled: boolean) {
    updateModalBusy = true;
    try {
      appSettings = await invoke<AppSettings>("save_app_settings", {
        settings: { autoCheckUpdates: enabled },
      });
    } catch (e) {
      console.error("Failed to save update settings:", e);
    } finally {
      updateModalBusy = false;
    }
  }

  async function skipCurrentUpdate() {
    if (!updateResult?.updateAvailable) return;

    updateModalBusy = true;
    try {
      appSettings = await invoke<AppSettings>("save_app_settings", {
        settings: { skippedUpdateVersion: updateResult.latestVersion },
      });
      updateModalOpen = false;
      updateCheckError = null;
      updateResult = null;
    } catch (e) {
      console.error("Failed to skip update version:", e);
    } finally {
      updateModalBusy = false;
    }
  }

  async function resumeUpdateReminders() {
    updateModalBusy = true;
    try {
      appSettings = await invoke<AppSettings>("save_app_settings", {
        settings: { skippedUpdateVersion: null },
      });
    } catch (e) {
      console.error("Failed to resume update reminders:", e);
    } finally {
      updateModalBusy = false;
    }
  }

  async function openUpdateUrl(url: string | null) {
    if (!url) return;
    try {
      await invoke("open_url", { url });
    } catch (e) {
      console.error("Failed to open update URL:", e);
    }
  }

  async function installUpdate() {
    if (!updateResult?.inAppInstallAvailable) return;

    updateModalBusy = true;
    updateActionError = null;
    try {
      await invoke("install_app_update");
    } catch (e) {
      updateActionError = formatError(e);
      console.error("Failed to install update:", e);
    } finally {
      updateModalBusy = false;
    }
  }

  function onTokenChange(configured: boolean) {
    hfTokenConfigured = configured;
  }

  function onMelodyflowFlashSettingUpdated(enabled: boolean) {
    appSettings = { ...appSettings, melodyflowUseFlashAttn: enabled };
  }

  function onCareyXlSettingUpdated(enabled: boolean) {
    appSettings = { ...appSettings, careyUseXlModels: enabled };
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
    let disposed = false;

    void (async () => {
      loadServices();
      checkToken();
      const settings = await loadAppSettings();

      if (!disposed && showAppUpdater && settings.autoCheckUpdates) {
        await runUpdateCheck({
          includeSkipped: false,
          openModalWhenCurrent: false,
          openModalOnError: false,
        });
      }
    })();

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
      disposed = true;
      clearInterval(pollTimer);
      unlisten.then((fn) => fn());
      unlistenSelect.then((fn) => fn());
      unlistenCloseRequest.then((fn) => fn());
    };
  });

  let runningCount = $derived(services.filter((s) => s.status === "running").length);
  let totalCount = $derived(services.length);
  let selectedService = $derived(services.find((s) => s.id === selectedServiceId) ?? null);
  let careyService = $derived(services.find((s) => s.id === "carey") ?? null);
</script>

<main>
  <header>
    <div class="header-left">
      <h1>gary4local</h1>
    </div>
    <div class="header-right">
      {#if showAppUpdater}
        <button
          class:accent={!!updateResult?.updateAvailable}
          onclick={checkForUpdatesManually}
          disabled={updateCheckBusy}
        >
          {#if updateCheckBusy}
            checking...
          {:else if updateResult?.updateAvailable}
            update {updateResult.latestVersion}
          {:else}
            check updates
          {/if}
        </button>
      {/if}
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
        onManageCareyLoras={showCareyLoras}
      />
    </div>
    <div class="divider"></div>
    <div class="right-panel">
      {#if rightPanel === "models" && modelServiceId}
        <ModelPanel serviceId={modelServiceId} onBack={backToLogs} />
      {:else}
        {#if selectedServiceId === "stable-audio"}
          <TokenBanner {onTokenChange} />
        {:else if selectedServiceId === "carey"}
          <CareyXlBanner
            enabled={appSettings.careyUseXlModels}
            serviceStatus={selectedService?.status ?? "stopped"}
            onUpdated={onCareyXlSettingUpdated}
          />
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
  <AppUpdateModal
    open={showAppUpdater && updateModalOpen}
    result={updateResult}
    error={updateCheckError}
    autoCheckEnabled={appSettings.autoCheckUpdates}
    isSkipped={appSettings.skippedUpdateVersion === updateResult?.latestVersion}
    busy={updateModalBusy}
    actionError={updateActionError}
    onClose={closeUpdateModal}
    onInstall={installUpdate}
    onDownload={() => openUpdateUrl(updateResult?.downloadUrl ?? null)}
    onSkipVersion={skipCurrentUpdate}
    onResumeReminders={resumeUpdateReminders}
    onAutoCheckChange={setAutoCheckUpdates}
  />
  <CareyLoraModal
    open={careyLoraModalOpen}
    serviceStatus={careyService?.status ?? "stopped"}
    serviceEnvExists={careyService?.env_exists ?? false}
    careyXlEnabled={appSettings.careyUseXlModels}
    onClose={closeCareyLoras}
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
    display: flex;
    align-items: center;
    gap: 12px;
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
