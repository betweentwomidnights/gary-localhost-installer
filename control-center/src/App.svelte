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
  import CareyScragVaeBanner from "./lib/CareyScragVaeBanner.svelte";
  import Sa3OutputPanel from "./lib/Sa3OutputPanel.svelte";
  import CareyLoraModal from "./lib/CareyLoraModal.svelte";
  import CareyAceTrainingModal from "./lib/CareyAceTrainingModal.svelte";
  import Sa3LoraModal from "./lib/Sa3LoraModal.svelte";
  import Sa3LoraTrainingModal from "./lib/Sa3LoraTrainingModal.svelte";
  import CloseBehaviorModal from "./lib/CloseBehaviorModal.svelte";
  import AppUpdateModal from "./lib/AppUpdateModal.svelte";
  import StorageSettingsModal from "./lib/StorageSettingsModal.svelte";

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

  interface Sa3LoudnessSettings {
    peakNormalizeDb: string;
    limiterCeilingDb: string;
    latentRescale: string;
    latentShift: string;
    latentTargetStd: string;
    continuationTailPad: string;
  }

  interface AppSettings {
    melodyflowUseFlashAttn: boolean;
    careyUseXlModels: boolean;
    careyUseScragVae: boolean;
    sa3Loudness: Sa3LoudnessSettings;
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

  interface RuntimeStorageInfo {
    activeRoot: string;
    configuredRoot: string | null;
    startupRoot: string;
    defaultRoot: string;
    legacyRoot: string;
    configPath: string;
    pendingRestart: boolean;
    usingLegacyDefault: boolean;
    defaultRootIsLegacy: boolean;
  }

  interface LegacyStorageCleanupItem {
    id: string;
    label: string;
    path: string;
    bytes: number;
  }

  interface LegacyLoraMigrationCandidate {
    service: string;
    name: string;
    sourcePath: string;
    targetPath: string;
    bytes: number;
  }

  interface LegacyStorageMaintenanceInfo {
    activeRoot: string;
    pendingRoot: string;
    legacyRoot: string;
    defaultHfCacheRoot: string;
    cleanupItems: LegacyStorageCleanupItem[];
    loraCandidates: LegacyLoraMigrationCandidate[];
    storageMoveLoraCandidates: LegacyLoraMigrationCandidate[];
    totalCleanupBytes: number;
    totalLoraBytes: number;
    totalStorageMoveLoraBytes: number;
    canCleanup: boolean;
    canMigrateLoras: boolean;
    canMigrateStorageLoras: boolean;
  }

  interface LegacyStorageMaintenanceResult {
    info: LegacyStorageMaintenanceInfo;
    migratedLoras: number;
    cleanedItems: number;
    warnings: string[];
    errors: string[];
  }

  interface RuntimeCacheInfo {
    uvCachePath: string;
    uvCacheBytes: number;
  }

  interface RuntimeCacheClearResult {
    info: RuntimeCacheInfo;
    clearedBytes: number;
  }

  interface ServiceEnvInfo {
    serviceId: string;
    displayName: string;
    envPath: string;
    envBytes: number;
    present: boolean;
    blockedReason: string | null;
  }

  interface ServiceEnvRemovalResult {
    serviceId: string;
    removedBytes: number;
    environments: ServiceEnvInfo[];
  }

  const showMelodyflowFlashBanner =
    import.meta.env.VITE_ENABLE_MELODYFLOW_FA2_TOGGLE !== "0";
  const showAppUpdater = import.meta.env.VITE_ENABLE_APP_UPDATER !== "0";

  let services: ServiceInfo[] = $state([]);
  let selectedServiceId: string | null = $state(null);
  let logText: string = $state("");
  let logViewerLive = $state(true);
  let pollTimer: number;
  let appSettings: AppSettings = $state({
    melodyflowUseFlashAttn: false,
    careyUseXlModels: false,
    careyUseScragVae: false,
    sa3Loudness: {
      peakNormalizeDb: "2.0",
      limiterCeilingDb: "-0.3",
      latentRescale: "1.0",
      latentShift: "0.0",
      latentTargetStd: "",
      continuationTailPad: "6",
    },
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
  let storageModalOpen = $state(false);
  let storageInfo: RuntimeStorageInfo | null = $state(null);
  let storageBusy = $state(false);
  let storageError: string | null = $state(null);
  let storageMaintenanceInfo: LegacyStorageMaintenanceInfo | null = $state(null);
  let storageMaintenanceBusy = $state(false);
  let storageMaintenanceError: string | null = $state(null);
  let storageMaintenanceWarning: string | null = $state(null);
  let storageMaintenanceMessage: string | null = $state(null);
  let runtimeCacheInfo: RuntimeCacheInfo | null = $state(null);
  let runtimeCacheBusy = $state(false);
  let runtimeCacheError: string | null = $state(null);
  let runtimeCacheMessage: string | null = $state(null);
  let serviceEnvs: ServiceEnvInfo[] | null = $state(null);
  let blobReclaimBusy = $state(false);
  let blobReclaimMessage: string | null = $state(null);
  let serviceEnvBusy: string | null = $state(null);
  let serviceEnvError: string | null = $state(null);
  let serviceEnvMessage: string | null = $state(null);
  let storageRestarting = $state(false);
  let careyLoraModalOpen = $state(false);
  let careyAceTrainingModalOpen = $state(false);
  let sa3LoraModalOpen = $state(false);
  let sa3LoraTrainingModalOpen = $state(false);

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

  async function fetchLog(serviceId: string, options: { force?: boolean } = {}) {
    if (!options.force && !logViewerLive) return;
    try {
      const nextLog = await invoke<string>("get_service_log", { serviceId });
      if (serviceId !== selectedServiceId) return;
      if (options.force || logViewerLive) {
        logText = nextLog;
      }
    } catch (e) {
      if (options.force || logViewerLive) {
        logText = `Error reading log: ${e}`;
      }
    }
  }

  function selectService(id: string) {
    selectedServiceId = id;
    logViewerLive = true;
    rightPanel = "logs";
    fetchLog(id, { force: true });
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

  function showCareyAceTraining() {
    selectedServiceId = "carey";
    careyAceTrainingModalOpen = true;
  }

  function closeCareyAceTraining() {
    careyAceTrainingModalOpen = false;
  }

  function showSa3Loras() {
    selectedServiceId = "sa3";
    sa3LoraModalOpen = true;
  }

  function closeSa3Loras() {
    sa3LoraModalOpen = false;
  }

  function showSa3LoraTraining() {
    selectedServiceId = "sa3";
    sa3LoraTrainingModalOpen = true;
  }

  function closeSa3LoraTraining() {
    sa3LoraTrainingModalOpen = false;
  }

  function backToLogs() {
    rightPanel = "logs";
    logViewerLive = true;
    if (selectedServiceId) fetchLog(selectedServiceId, { force: true });
  }

  function onLogLiveUpdatesChange(live: boolean) {
    logViewerLive = live;
    if (live && selectedServiceId && rightPanel === "logs") {
      fetchLog(selectedServiceId, { force: true });
    }
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

  async function loadRuntimeStorageInfo() {
    try {
      storageInfo = await invoke<RuntimeStorageInfo>("get_runtime_storage_info");
      storageError = null;
    } catch (e) {
      storageError = formatError(e);
    }
    return storageInfo;
  }

  async function loadStorageMaintenanceInfo() {
    try {
      storageMaintenanceInfo = await invoke<LegacyStorageMaintenanceInfo>(
        "get_legacy_storage_maintenance_info",
      );
      storageMaintenanceError = null;
    } catch (e) {
      storageMaintenanceError = formatError(e);
    }
    return storageMaintenanceInfo;
  }

  async function loadRuntimeCacheInfo() {
    try {
      runtimeCacheInfo = await invoke<RuntimeCacheInfo>("get_runtime_cache_info");
      runtimeCacheError = null;
    } catch (e) {
      runtimeCacheError = formatError(e);
    }
    return runtimeCacheInfo;
  }

  function formatError(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }

  function formatByteCount(bytes: number): string {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
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

  async function openStorageSettings() {
    storageModalOpen = true;
    await Promise.all([
      loadRuntimeStorageInfo(),
      loadStorageMaintenanceInfo(),
      loadRuntimeCacheInfo(),
      loadServiceEnvs(),
    ]);
  }

  function closeStorageSettings() {
    storageModalOpen = false;
    storageError = null;
    storageMaintenanceError = null;
    storageMaintenanceWarning = null;
    storageMaintenanceMessage = null;
    runtimeCacheError = null;
    runtimeCacheMessage = null;
    serviceEnvError = null;
    serviceEnvMessage = null;
    blobReclaimMessage = null;
    // serviceEnvs is kept so reopening shows the last known sizes straight
    // away while the rescan runs behind it.
  }

  async function clearRuntimeUvCache() {
    runtimeCacheBusy = true;
    runtimeCacheError = null;
    runtimeCacheMessage = null;
    try {
      const result = await invoke<RuntimeCacheClearResult>("clear_uv_cache");
      runtimeCacheInfo = result.info;
      runtimeCacheMessage = result.clearedBytes > 0
        ? `cleared ${formatByteCount(result.clearedBytes)} from the UV cache`
        : "the UV cache was already empty";
    } catch (e) {
      runtimeCacheError = formatError(e);
    } finally {
      runtimeCacheBusy = false;
    }
  }

  async function loadServiceEnvs() {
    try {
      serviceEnvs = await invoke<ServiceEnvInfo[]>("get_service_envs");
      serviceEnvError = null;
    } catch (e) {
      serviceEnvError = formatError(e);
    }
    return serviceEnvs;
  }

  async function removeServiceEnv(serviceId: string) {
    serviceEnvBusy = serviceId;
    serviceEnvError = null;
    serviceEnvMessage = null;
    try {
      const result = await invoke<ServiceEnvRemovalResult>("remove_service_env", { serviceId });
      serviceEnvs = result.environments;
      const label = result.serviceId;
      serviceEnvMessage = result.removedBytes > 0
        ? `removed ${label}'s environment (${formatByteCount(result.removedBytes)}) - rebuild it when you next need that model`
        : `${label} had no environment installed`;
    } catch (e) {
      serviceEnvError = formatError(e);
    } finally {
      serviceEnvBusy = null;
    }
  }

  async function reclaimDuplicateBlobs() {
    blobReclaimBusy = true;
    blobReclaimMessage = null;
    serviceEnvError = null;
    try {
      const freed = await invoke<number>("reclaim_duplicate_blobs");
      blobReclaimMessage = freed > 0
        ? `freed ${formatByteCount(freed)} of duplicated copies`
        : "no duplicates found - nothing to clean up";
    } catch (e) {
      serviceEnvError = formatError(e);
    } finally {
      blobReclaimBusy = false;
    }
  }

  async function saveRuntimeStorageRoot(path: string) {
    storageBusy = true;
    storageError = null;
    try {
      storageInfo = await invoke<RuntimeStorageInfo>("save_runtime_storage_root", { path });
      await loadStorageMaintenanceInfo();
    } catch (e) {
      storageError = formatError(e);
    } finally {
      storageBusy = false;
    }
  }

  async function resetRuntimeStorageRoot() {
    storageBusy = true;
    storageError = null;
    try {
      storageInfo = await invoke<RuntimeStorageInfo>("reset_runtime_storage_root");
      await loadStorageMaintenanceInfo();
    } catch (e) {
      storageError = formatError(e);
    } finally {
      storageBusy = false;
    }
  }

  async function refreshStorageMaintenance() {
    storageMaintenanceBusy = true;
    storageMaintenanceError = null;
    storageMaintenanceWarning = null;
    storageMaintenanceMessage = null;
    try {
      await loadStorageMaintenanceInfo();
    } finally {
      storageMaintenanceBusy = false;
    }
  }

  async function migrateLegacyLoras() {
    storageMaintenanceBusy = true;
    storageMaintenanceError = null;
    storageMaintenanceWarning = null;
    storageMaintenanceMessage = null;
    try {
      const result = await invoke<LegacyStorageMaintenanceResult>("migrate_legacy_loras");
      storageMaintenanceInfo = result.info;
      storageMaintenanceMessage =
        result.migratedLoras > 0
          ? `copied ${result.migratedLoras} LoRA${result.migratedLoras === 1 ? "" : "s"}`
          : "no LoRAs needed copying";
      if (result.errors.length > 0) {
        storageMaintenanceError = result.errors.join("\n");
      }
      if (result.warnings.length > 0) {
        storageMaintenanceWarning = result.warnings.join("\n");
      }
    } catch (e) {
      storageMaintenanceError = formatError(e);
    } finally {
      storageMaintenanceBusy = false;
    }
  }

  async function migrateStorageLorasToPendingRoot() {
    storageMaintenanceBusy = true;
    storageMaintenanceError = null;
    storageMaintenanceWarning = null;
    storageMaintenanceMessage = null;
    try {
      const result = await invoke<LegacyStorageMaintenanceResult>(
        "migrate_storage_loras_to_pending_root",
      );
      storageMaintenanceInfo = result.info;
      storageMaintenanceMessage =
        result.migratedLoras > 0
          ? `copied ${result.migratedLoras} LoRA${result.migratedLoras === 1 ? "" : "s"} to next storage`
          : "no LoRAs needed copying";
      if (result.errors.length > 0) {
        storageMaintenanceError = result.errors.join("\n");
      }
      if (result.warnings.length > 0) {
        storageMaintenanceWarning = result.warnings.join("\n");
      }
    } catch (e) {
      storageMaintenanceError = formatError(e);
    } finally {
      storageMaintenanceBusy = false;
    }
  }

  async function cleanupLegacyStorage() {
    storageMaintenanceBusy = true;
    storageMaintenanceError = null;
    storageMaintenanceWarning = null;
    storageMaintenanceMessage = null;
    try {
      const result = await invoke<LegacyStorageMaintenanceResult>("cleanup_legacy_storage");
      storageMaintenanceInfo = result.info;
      storageMaintenanceMessage =
        result.cleanedItems > 0
          ? `cleaned ${result.cleanedItems} old item${result.cleanedItems === 1 ? "" : "s"}`
          : "nothing old needed cleanup";
      if (result.errors.length > 0) {
        storageMaintenanceError = result.errors.join("\n");
      }
      if (result.warnings.length > 0) {
        storageMaintenanceWarning = result.warnings.join("\n");
      }
    } catch (e) {
      storageMaintenanceError = formatError(e);
    } finally {
      storageMaintenanceBusy = false;
    }
  }

  async function restartApplication() {
    storageRestarting = true;
    storageError = null;
    try {
      await invoke("restart_application");
    } catch (e) {
      storageRestarting = false;
      storageError = formatError(e);
    }
  }

  async function revealStoragePath(path: string) {
    try {
      await invoke("reveal_path", { path });
    } catch (e) {
      storageError = formatError(e);
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

  function onCareyScragVaeSettingUpdated(enabled: boolean) {
    appSettings = { ...appSettings, careyUseScragVae: enabled };
  }

  function onSa3LoudnessSettingUpdated(settings: Sa3LoudnessSettings) {
    appSettings = { ...appSettings, sa3Loudness: settings };
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
      loadRuntimeStorageInfo();
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
      if (selectedServiceId && rightPanel === "logs" && logViewerLive) fetchLog(selectedServiceId);
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
  let sa3Service = $derived(services.find((s) => s.id === "sa3") ?? null);
</script>

<main>
  <header>
    <div class="header-left">
      <h1>gary4local-rocm</h1>
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
      <button
        class:accent={storageInfo?.pendingRestart}
        onclick={openStorageSettings}
        disabled={storageBusy}
        title={storageInfo?.activeRoot ?? "runtime storage"}
      >
        storage
      </button>
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
        onTrainCareyAce={showCareyAceTraining}
        onManageSa3Loras={showSa3Loras}
        onTrainSa3Lora={showSa3LoraTraining}
      />
    </div>
    <div class="divider"></div>
    <div class="right-panel">
      {#if rightPanel === "models" && modelServiceId}
        <ModelPanel serviceId={modelServiceId} onBack={backToLogs} />
      {:else}
        {#if selectedServiceId === "stable-audio" || selectedServiceId === "sa3"}
          <TokenBanner serviceId={selectedServiceId ?? "stable-audio"} {onTokenChange} />
          {#if selectedServiceId === "sa3"}
            <Sa3OutputPanel
              settings={appSettings.sa3Loudness}
              serviceStatus={selectedService?.status ?? "stopped"}
              onUpdated={onSa3LoudnessSettingUpdated}
            />
          {/if}
        {:else if selectedServiceId === "carey"}
          <CareyXlBanner
            enabled={appSettings.careyUseXlModels}
            serviceStatus={selectedService?.status ?? "stopped"}
            onUpdated={onCareyXlSettingUpdated}
          />
          <CareyScragVaeBanner
            enabled={appSettings.careyUseScragVae}
            serviceStatus={selectedService?.status ?? "stopped"}
            onUpdated={onCareyScragVaeSettingUpdated}
            onShowModels={() => showModels("carey")}
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
          onLiveUpdatesChange={onLogLiveUpdatesChange}
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
  <StorageSettingsModal
    open={storageModalOpen}
    info={storageInfo}
    busy={storageBusy}
    error={storageError}
    maintenanceInfo={storageMaintenanceInfo}
    maintenanceBusy={storageMaintenanceBusy}
    maintenanceError={storageMaintenanceError}
    maintenanceWarning={storageMaintenanceWarning}
    maintenanceMessage={storageMaintenanceMessage}
    cacheInfo={runtimeCacheInfo}
    cacheBusy={runtimeCacheBusy}
    cacheError={runtimeCacheError}
    cacheMessage={runtimeCacheMessage}
    serviceEnvs={serviceEnvs}
    serviceEnvBusy={serviceEnvBusy}
    serviceEnvError={serviceEnvError}
    serviceEnvMessage={serviceEnvMessage}
    onRemoveServiceEnv={removeServiceEnv}
    blobReclaimBusy={blobReclaimBusy}
    blobReclaimMessage={blobReclaimMessage}
    onReclaimBlobs={reclaimDuplicateBlobs}
    restarting={storageRestarting}
    onChoose={saveRuntimeStorageRoot}
    onReset={resetRuntimeStorageRoot}
    onReveal={revealStoragePath}
    onRefreshMaintenance={refreshStorageMaintenance}
    onMigrateLoras={migrateLegacyLoras}
    onMigrateStorageLoras={migrateStorageLorasToPendingRoot}
    onCleanupLegacy={cleanupLegacyStorage}
    onClearUvCache={clearRuntimeUvCache}
    onRestart={restartApplication}
    onClose={closeStorageSettings}
  />
  <CareyLoraModal
    open={careyLoraModalOpen}
    serviceStatus={careyService?.status ?? "stopped"}
    serviceEnvExists={careyService?.env_exists ?? false}
    careyXlEnabled={appSettings.careyUseXlModels}
    onClose={closeCareyLoras}
  />
  <CareyAceTrainingModal
    open={careyAceTrainingModalOpen}
    serviceStatus={careyService?.status ?? "stopped"}
    serviceEnvExists={careyService?.env_exists ?? false}
    onClose={closeCareyAceTraining}
    onShowModels={() => {
      // The models panel lives in the main window behind this modal, so
      // sending someone there without closing it looks like nothing happened.
      closeCareyAceTraining();
      showModels("carey");
    }}
  />
  <Sa3LoraModal
    open={sa3LoraModalOpen}
    serviceStatus={sa3Service?.status ?? "stopped"}
    serviceEnvExists={sa3Service?.env_exists ?? false}
    onClose={closeSa3Loras}
  />
  <Sa3LoraTrainingModal
    open={sa3LoraTrainingModalOpen}
    serviceStatus={sa3Service?.status ?? "stopped"}
    serviceEnvExists={sa3Service?.env_exists ?? false}
    onClose={closeSa3LoraTraining}
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
