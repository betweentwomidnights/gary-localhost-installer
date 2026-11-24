import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runtimeDevice } from '../src/lib/nativeDevice.ts';

test('Metal recommendations use the MTL device and ignore the CPU', () => {
  const info = {
    backend: 'metal',
    devices: [
      { name: 'Apple M4', backend: 'MTL', deviceType: 'gpu', memoryTotalBytes: 24e9 },
      { name: 'Apple M4 CPU', backend: 'CPU', deviceType: 'cpu', memoryTotalBytes: 32e9 },
    ],
  };
  assert.equal(runtimeDevice(info)?.memoryTotalBytes, 24e9);
});

test('CUDA still uses its dedicated GPU rather than a larger Vulkan device', () => {
  const info = {
    backend: 'cuda',
    devices: [
      { name: 'RTX 5070', backend: 'CUDA', deviceType: 'gpu', memoryTotalBytes: 8e9 },
      { name: 'Integrated', backend: 'Vulkan', deviceType: 'igpu', memoryTotalBytes: 32e9 },
    ],
  };
  assert.equal(runtimeDevice(info)?.name, 'RTX 5070');
});
