const STORAGE_PREFIX = "gary4local.dialog-directory.";

export function rememberedDialogDirectory(key: string): string | undefined {
  try {
    return window.localStorage.getItem(`${STORAGE_PREFIX}${key}`) || undefined;
  } catch {
    return undefined;
  }
}

export function rememberDialogSelection(
  key: string,
  selectedPath: string,
  selectionType: "file" | "directory",
): void {
  let directory = selectedPath;
  if (selectionType === "file") {
    const separatorIndex = Math.max(selectedPath.lastIndexOf("\\"), selectedPath.lastIndexOf("/"));
    if (separatorIndex < 0) return;
    directory = separatorIndex === 0
      ? selectedPath.slice(0, 1)
      : separatorIndex === 2 && /^[a-z]:/i.test(selectedPath)
        ? selectedPath.slice(0, 3)
        : selectedPath.slice(0, separatorIndex);
  }
  if (!directory) return;

  try {
    window.localStorage.setItem(`${STORAGE_PREFIX}${key}`, directory);
  } catch {
    // Picker memory is a convenience; storage restrictions should not block the dialog.
  }
}
