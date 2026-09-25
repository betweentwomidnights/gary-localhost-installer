import { invoke } from "@tauri-apps/api/core";

export interface YueyTiers {
  installed: string[];
  chosen: string;
  active: string | null;
}

export function loadYueyTiers(): Promise<YueyTiers> {
  return invoke<YueyTiers>("get_yuey_tiers");
}

// Save the tier yuey should launch with ("" for automatic) and restart it if
// it is running, since the tier is fixed at launch. Returns what happened.
export async function chooseYueyTier(encoding: string): Promise<string> {
  await invoke("save_app_settings", { settings: { yueyEncoding: encoding } });
  const tiers = await loadYueyTiers();
  const services = await invoke<{ id: string; status: string }[]>("get_services");
  const status = services.find((service) => service.id === "yuey")?.status;
  const target = tiers.active ?? "no tier";
  if (status === "running" || status === "starting" || status === "unhealthy") {
    await invoke("restart_service", { serviceId: "yuey" });
    return `yuey is restarting on ${target}.`;
  }
  return `yuey will use ${target} next time it starts.`;
}
