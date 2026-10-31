export interface NativeDeviceInfo {
  backend: string | null;
  devices: { name: string; backend: string; deviceType: string; memoryTotalBytes: number }[];
  selectedDeviceName?: string | null;
  selectedMemoryBytes?: number | null;
}

// The GPU a native runtime runs on, as gary4local recommends models for it:
// the device the server's probe selected, else the largest dedicated GPU of
// its backend, else an integrated one. An integrated GPU reports shared system
// memory, so it must not stand in for a dedicated card beside it.
export function runtimeDevice(info: NativeDeviceInfo | null): { name: string; memoryTotalBytes: number } | undefined {
  if (!info) return undefined;
  if (info.selectedDeviceName && info.selectedMemoryBytes) {
    return { name: info.selectedDeviceName, memoryTotalBytes: info.selectedMemoryBytes };
  }
  const backend = (info.backend ?? "").toLowerCase();
  const largest = (kind: string) => info.devices
    .filter((device) => device.deviceType.toLowerCase() === kind && device.backend.toLowerCase().includes(backend))
    .sort((left, right) => right.memoryTotalBytes - left.memoryTotalBytes)[0];
  return largest("gpu") ?? largest("igpu");
}
