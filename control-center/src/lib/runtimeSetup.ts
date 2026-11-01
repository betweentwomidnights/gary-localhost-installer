import { invoke } from "@tauri-apps/api/core";
import { recommendQuantizedSa3Training, type Sa3TrainingHardware } from "./sa3NativeTraining";

export interface Item { label: string; path: string; bytes: number; kind: string | null; }
export interface Selection { encoding: string; verifiedRelease: string; cleanupComplete: boolean; cleanupErrors: string[]; }
export interface Preview {
  activeRoot: string;
  hfHubCache: string;
  cleanupCandidates: Item[];
  legacyRuntimePresent: boolean;
  trainingBasePresent: boolean;
  estimatedCleanupBytes: number;
  preservedPaths: Item[];
  warnings: string[];
  nativeSelection: Selection | null;
  cleanupToken: string;
  pythonStillNeeded: string | null;
}
export interface Profile {
  serviceId: string;
  label: string;
  dataLabel: string | null;
  tiers: string[];
  tierComponents: Record<string, string[]>;
  features: string[];
  tierOptions: Record<string, string[]>;
  optionsLabel: string | null;
}
export interface Component { id: string; label: string; files: { bytes: number }[]; }
export interface Recommendations { tier: string | null; components: string[]; }
export type SetupTask = "runtime" | "models" | "adapters" | "verify";

/** What a setup prepares: a precision, the extra models picked for it, and SA3's extras. */
export interface SetupChoices { encoding: string; options: string[]; training: string | null; decoder: boolean; }

export const hasFeature = (profile: Profile | null, feature: string) => profile?.features.includes(feature) ?? false;

export function componentIds(profile: Profile, choices: SetupChoices): string[] {
  return [
    ...(profile.tierComponents[choices.encoding] ?? []),
    ...choices.options,
    ...(choices.training ? [`sa3-native::medium-base-${choices.training}`] : []),
    ...(choices.decoder ? ["sa3-native::decoder-correction"] : []),
  ];
}

export const checksLabel = (choices: SetupChoices) =>
  choices.training ? "generation, training and checkpoint saving" : "generation";

/** Nothing of the user's is left to review: no Python environment or weights,
 *  only unchanged bundled Python source, which setup removes by itself. */
export function onlyBundledCode(preview: Preview) {
  return !preview.legacyRuntimePresent
    && !preview.warnings.length
    && !preview.pythonStillNeeded
    && preview.cleanupCandidates.every((item) => item.kind === "code");
}

/** The service's setup details and the choices the setup modal starts from. */
export async function loadSetup(serviceId: string) {
  const [profile, preview, catalog, recommended] = await Promise.all([
    invoke<Profile>("get_migration_profile", { serviceId }),
    invoke<Preview>("get_migration_preview", { serviceId }),
    invoke<Component[]>("get_native_model_catalog", { serviceId }),
    invoke<Recommendations>("get_native_model_recommendations", { serviceId }),
  ]);
  let training: string | null = null;
  if (hasFeature(profile, "training-base") && preview.trainingBasePresent) {
    const hardware = await invoke<Sa3TrainingHardware>("get_native_runtime_info", { serviceId });
    training = recommendQuantizedSa3Training(hardware) ? "Q4_K_M" : "F16";
  }
  const decoder = hasFeature(profile, "decoder-correction")
    && (await invoke<{ enabled: boolean }>("get_sa3_native_decoder_state")).enabled;
  const choices: SetupChoices = {
    encoding: preview.nativeSelection?.encoding ?? recommended.tier ?? profile.tiers[0],
    options: [],
    training,
    decoder,
  };
  return { profile, preview, catalog, recommended, choices };
}

export async function stopService(serviceId: string) {
  const current = await invoke<{ id: string; status: string }[]>("get_services");
  if (current.some((entry) => entry.id === serviceId && ["running", "starting", "unhealthy"].includes(entry.status))) {
    await invoke("stop_service", { serviceId });
  }
}

/** Install the runtime, prepare the models, then prove the runtime and switch
 *  to it. `settle` gets the recommendations measured by the runtime install
 *  and returns the final choices. */
export async function runSetup(
  serviceId: string,
  profile: Profile,
  settle: (recommended: Recommendations) => SetupChoices,
  report: (task: SetupTask, message: string) => void,
) {
  report("runtime", `Stopping ${profile.label}...`);
  await stopService(serviceId);
  report("runtime", "Installing and checking the runtime for your hardware...");
  await invoke("prepare_native_runtime", { serviceId });
  const choices = settle(await invoke<Recommendations>("get_native_model_recommendations", { serviceId }));
  report("models", "Downloading and verifying model files. Existing components are reused.");
  await invoke("prepare_native_models", { serviceId, componentIds: componentIds(profile, choices), waitForCompletion: true });
  if (choices.decoder) {
    report("models", "Preparing your decoder squeak fix...");
    await invoke("prepare_sa3_native_decoder");
  }
  if (hasFeature(profile, "adapters")) {
    report("adapters", "Preparing native copies of your registered LoRAs...");
    const loras = await invoke<{ entries: { name: string; nativePath: string | null; error: string | null }[] }>("prepare_sa3_native_loras");
    const failed = loras.entries.filter((entry) => !entry.nativePath || entry.error);
    if (failed.length) throw new Error(failed.map((entry) => `${entry.name}: ${entry.error ?? "Native copy is missing"}`).join("\n"));
  }
  report("verify", `Verifying the server, model files and native ${checksLabel(choices)}...`);
  await invoke<Selection>("activate_native_runtime", { serviceId, encoding: choices.encoding });
}

/** Remove the reviewed files `preview` lists. Returns what could not be removed. */
export async function cleanUp(serviceId: string, preview: Preview) {
  // The review token binds this to exactly the paths the preview listed.
  return invoke<{ selection: Selection; errors: string[] }>("cleanup_legacy_installation", { serviceId, reviewToken: preview.cleanupToken });
}

/** Set up a service with the defaults its setup modal would use, finishing
 *  without review when there is no Python installation to clean up. */
export async function setUpWithDefaults(serviceId: string, report: (task: SetupTask, message: string) => void) {
  const setup = await loadSetup(serviceId);
  await runSetup(
    serviceId,
    setup.profile,
    (recommended) => setup.preview.nativeSelection || !recommended.tier ? setup.choices : { ...setup.choices, encoding: recommended.tier },
    report,
  );
  const preview = await invoke<Preview>("get_migration_preview", { serviceId });
  if (!onlyBundledCode(preview)) return;
  report("verify", "Checking again and removing unused bundled Python files...");
  const result = await cleanUp(serviceId, preview);
  if (result.errors.length) throw new Error(result.errors.join("\n"));
}
