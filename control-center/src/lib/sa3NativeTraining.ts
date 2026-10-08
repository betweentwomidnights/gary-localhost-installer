export interface Sa3TrainingHardware {
  backend: string | null;
  devices: { backend: string; deviceType: string; memoryTotalBytes: number }[];
}

export function recommendQuantizedSa3Training(info: Sa3TrainingHardware): boolean {
  const memory = info.devices.find((device) => device.backend.toLowerCase() === "cuda" && device.deviceType.toLowerCase() === "gpu")?.memoryTotalBytes ?? 0;
  return info.backend === "cuda" && memory > 0 && memory <= 9 * 1024 ** 3;
}
