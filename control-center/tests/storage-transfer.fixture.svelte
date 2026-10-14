<script lang="ts">
  import Storage from '../src/lib/StorageSettingsModal.svelte';
  const root = 'C:/Users/Gary/AppData/Roaming/Gary4JUCE';
  const noop = () => {};
  let busy = $state(false);
  let error = $state<string | null>(null);
  let progress = $state<{message:string;completedBytes:number;totalBytes:number}|null>(null);
  let info = $state({activeRoot:root,configuredRoot:null as string|null,startupRoot:root,defaultRoot:root,legacyRoot:root,configPath:'C:/Users/Gary/AppData/Local/com.betweentwomidnights.gary4local/storage.json',pendingRestart:false,usingLegacyDefault:true,defaultRootIsLegacy:true});
  async function choose(path:string) {
    busy = true;
    error = null;
    progress = {message:'Checking native runtimes, models and LoRA history…',completedBytes:0,totalBytes:0};
    await new Promise(resolve=>setTimeout(resolve,300));
    progress = {message:'Copying models/sa3/stable-audio-3-medium-base-dit-1.5B-v1.0-F16.gguf',completedBytes:40,totalBytes:100};
    await new Promise(resolve=>setTimeout(resolve,new URLSearchParams(location.search).has('hold') ? 60000 : 8000));
    if (new URLSearchParams(location.search).has('error')) {
      error = 'Not enough free space in the selected folder. Current storage is unchanged.';
    } else {
      info = {...info,configuredRoot:path,startupRoot:path,pendingRestart:true};
      progress = {message:'Native runtimes, models and LoRAs copied and verified. Restart to use the new storage.',completedBytes:100,totalBytes:100};
    }
    busy = false;
  }
</script>
<Storage open={true} {info} {busy} {error} transferProgress={progress}
  maintenanceInfo={null} maintenanceError={null} maintenanceWarning={null} maintenanceMessage={null}
  cacheInfo={null} cacheError={null} cacheMessage={null}
  serviceEnvError={null} serviceEnvMessage={null} blobReclaimMessage={null}
  onChoose={choose} onReset={()=>choose(root)} onReveal={noop} onRefreshMaintenance={noop}
  onMigrateLoras={noop} onMigrateStorageLoras={noop} onCleanupLegacy={noop}
  onClearUvCache={noop} onRemoveServiceEnv={noop} onReclaimBlobs={noop} onRestart={noop} onClose={noop}/>
<div class="fixture-label">Simulated storage transfer — no runtime files change.</div>
<style>.fixture-label{position:fixed;bottom:8px;left:8px;z-index:10000;color:var(--text-secondary);font:11px var(--font-mono);pointer-events:none}</style>
