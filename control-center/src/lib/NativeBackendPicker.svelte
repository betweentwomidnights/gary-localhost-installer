<script lang="ts">
  let { value, backends, disabled = false, onChange }: {
    value: string;
    backends: string[];
    disabled?: boolean;
    onChange: (backend: string) => void;
  } = $props();
</script>

<label class="backend">
  <span>backend</span>
  <select {value} {disabled} onchange={(event) => onChange(event.currentTarget.value)}>
    <option value="auto">auto</option>
    {#each backends as backend}<option value={backend}>{backend}</option>{/each}
  </select>
</label>

<style>
  .backend { display: inline-flex; align-items: center; gap: 8px; font-size: 11px; color: var(--text-primary); white-space: nowrap; }
  select { font-size: 11px; }
</style>
