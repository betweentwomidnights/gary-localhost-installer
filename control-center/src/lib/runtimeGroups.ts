// Names for native runtimes that several services share, by manifest bundle.
const labels: Record<string, string> = { sa3: "stable audio", audiocraft: "audiocraft" };

export function runtimeGroupLabel(group: string): string {
  return labels[group] ?? group;
}
