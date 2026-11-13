import { mount } from 'svelte';
import { mockIPC } from '@tauri-apps/api/mocks';
import { emit } from '@tauri-apps/api/event';
import '../../src/styles.css';
import Harness from './Harness.svelte';
import sa3 from '../../src-tauri/sa3-models.json';
import carey from '../../src-tauri/carey-models.json';
import gary from '../../src-tauri/gary-models.json';
import jerry from '../../src-tauri/stable-audio-models.json';
import terry from '../../src-tauri/melodyflow-models.json';
import foundation from '../../src-tauri/foundation-models.json';

const query = new URLSearchParams(location.search);
const serviceId = query.get('service') ?? 'sa3';
const reference = serviceId === 'carey' ? 'BF16' : 'F16';
const tiers = serviceId === 'melodyflow' ? ['F16'] : [reference, 'Q8_0', 'Q4_K_M'];
const catalogs = { sa3, carey, gary, 'stable-audio': jerry, melodyflow: terry, foundation };
const catalog = catalogs[serviceId] ?? [];
const primaryId = (tier) => serviceId === 'sa3' ? `sa3-native::medium-${tier}`
  : serviceId === 'carey' ? `carey-native::base::${tier}`
  : serviceId === 'gary' ? `gary-native::vanya_ai_dnb_0.1::${tier}`
  : `${serviceId}-native::${tier}`;
const model = (id, label, category = 'native') => ({
  id, display_name: label, service: serviceId, size_category: category, group: null, epoch: null,
  status: query.has('all') || !id.includes('Q') ? 'downloaded' : 'available', downloaded_bytes: 1024,
});
let models = serviceId === 'yuey'
  ? [model('yuey::shared', 'YuE shared components', 'shared'), ...tiers.map((tier) => model(`yuey::${tier}`, `YuE ${tier}`, 'tier'))]
  : catalog.map((entry) => model(entry.id, entry.label));
if (query.has('missing-shared')) models.filter((entry) => entry.id.endsWith('::text') || entry.id.endsWith('::shared') || entry.id.endsWith('::encoder')).forEach((entry) => entry.status = 'available');
if (query.has('failed')) models.find((entry) => entry.id === primaryId('Q4_K_M')).status = 'failed';
let selected = reference;
const downloads = [];
window.testCalls = [];
mockIPC(async (command, args) => {
  window.testCalls.push({ command, args });
  switch (command) {
    case 'get_models': return models;
    case 'get_download_progress': return downloads;
    case 'get_native_model_catalog': return catalog;
    case 'get_native_runtime_selection': return { encoding: selected };
    case 'get_native_model_recommendations': return { tier: reference, components: [primaryId(reference)] };
    case 'get_native_runtime_info': return { installed: true, devOverride: false, backend: 'vulkan', preference: 'auto', offeredBackends: ['vulkan'], version: 'test', manifestVersion: 'test', devices: [], recommendedEncoding: reference };
    case 'get_sa3_native_decoder_state': return { enabled: false, prepared: false, sourcePresent: false, error: null };
    case 'get_migration_profile': return { label: serviceId, tiers,
      tierComponents: Object.fromEntries(tiers.map((tier) => [tier, [primaryId(tier), ...(serviceId === 'foundation' ? ['foundation-native::encoder'] : [])]])),
      tierOptions: {}, optionsLabel: null };
    case 'get_yuey_tiers': return { installed: tiers.filter((tier) => models.find((entry) => entry.id === `yuey::${tier}`)?.status === 'downloaded'), chosen: selected, active: selected };
    case 'get_services': return [{ id: serviceId, runtime: 'native', python_retired: true, status: query.has('running') ? 'running' : 'stopped' }];
    case 'select_native_model': selected = args.encoding; return { encoding: selected };
    case 'save_app_settings': selected = args.settings.yueyEncoding ?? selected; return {};
    case 'restart_service': return;
    case 'download_model': {
      models = models.map((entry) => entry.id === args.modelId ? { ...entry, status: 'downloading' } : entry);
      downloads.push({ model_id: args.modelId, progress: 0.4, status: 'downloading', message: 'Downloading model', error: null });
      await emit('models-updated', models);
      await emit('download-progress', downloads);
      return;
    }
    default: throw new Error(`Unexpected test IPC: ${command}`);
  }
}, { shouldMockEvents: true });
mount(Harness, { target: document.getElementById('app'), props: { serviceId, controls: query.has('controls') } });
