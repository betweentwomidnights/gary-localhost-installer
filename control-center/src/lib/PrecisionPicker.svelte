<script lang="ts">
  import type { PrecisionOption } from "./modelPrecision";

  let { label, value, options, active = null, recommended = null, disabled = false, onChange }: {
    label: string;
    value: string | undefined;
    options: PrecisionOption[];
    active?: string | null;
    recommended?: string | null;
    disabled?: boolean;
    onChange: (value: string) => void;
  } = $props();
</script>

<select aria-label={`${label} precision`} {value} {disabled} onchange={(event) => onChange(event.currentTarget.value)}>
  {#each options as option (option.value)}
    <option value={option.value}>{option.value}{option.value === active ? " · selected" : ""}{option.status === "downloaded" ? " · on disk" : option.status === "downloading" ? " · downloading" : ""}{option.value === recommended ? " · recommended" : ""}</option>
  {/each}
</select>

<style>
  select { font-size: 11px; padding: 2px 4px; max-width: 100%; }
</style>
