export interface PrecisionOption {
  value: string;
  status?: string;
}

// Browsing a copy never activates it. Keep an in-flight download visible,
// then prefer the user's choice, the saved selection, and an installed copy.
export function shownPrecision(
  options: PrecisionOption[],
  picked: string | null | undefined,
  active: string | null | undefined,
  recommended?: string | null,
): string | undefined {
  const has = (value: string | null | undefined) => options.some((option) => option.value === value);
  return options.find((option) => option.status === "downloading")?.value
    ?? (has(picked) ? picked! : undefined)
    ?? (has(active) ? active! : undefined)
    ?? options.find((option) => option.status === "downloaded")?.value
    ?? (has(recommended) ? recommended! : undefined)
    ?? options[0]?.value;
}
