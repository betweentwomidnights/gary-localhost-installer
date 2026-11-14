<script>
  import ModelPanel from '../../src/lib/ModelPanel.svelte';
  import YueyGenerationPanel from '../../src/lib/YueyGenerationPanel.svelte';
  import NativeRuntimeBanner from '../../src/lib/NativeRuntimeBanner.svelte';
  import StorageSettingsModal from '../../src/lib/StorageSettingsModal.svelte';
  import ServiceRow from '../../src/lib/ServiceRow.svelte';
  import { onMount } from 'svelte';
  import { listen } from '@tauri-apps/api/event';
  let { serviceId, controls, showRow, initialService } = $props();
  let service = $state(initialService);
  onMount(() => {
    const listener = listen('services-updated', (event) => service = event.payload[0]);
    return () => { void listener.then((stop) => stop()); };
  });
  let storageOpen = $state(true);
  const noop = () => {};
  const root = '/home/kev/gary4local-packaging/desktop-gui/data';
  const info = { activeRoot: root, startupRoot: root, defaultRoot: root, legacyRoot: root,
    configuredRoot: null, configPath: `${root}/settings.json`, pendingRestart: false, usingLegacyDefault: false, defaultRootIsLegacy: false };
</script>

{#if controls}
  <YueyGenerationPanel settings={{ instrumentalMethod: 'transfer', instrumentalAdapter: true, naturalMaxSeconds: 96 }} serviceStatus="stopped" onUpdated={noop} />
  <NativeRuntimeBanner {serviceId} label="SA3" serviceStatus="stopped" nativeBackend="vulkan" building={false} />
  <StorageSettingsModal open={storageOpen} {info} error={null} onClose={() => storageOpen = false} onReveal={noop}
    serviceEnvs={['audiocraft shared runtime', 'stable audio shared runtime'].map((displayName, i) => ({ serviceId: String(i), displayName, kind: 'native', envPath: root, envBytes: 1234567890, present: true, blockedReason: null }))} />
{:else}
  {#if showRow}<ServiceRow {service} selected={false} onSelect={noop} hasModels={true} hasMigration={true} />{/if}
  <ModelPanel {serviceId} serviceRuntime={service.runtime} hasMigration={serviceId !== 'yuey'} pythonRetired={service.python_retired} onBack={noop} />
{/if}

<style>
  :global(#app) { height: 100%; display: flex; flex-direction: column; }
</style>
