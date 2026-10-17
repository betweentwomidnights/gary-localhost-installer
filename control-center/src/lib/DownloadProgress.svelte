<script lang="ts">
  // A model download's progress, drawn like the Models panel's own bars.
  let { value, label }: { value: number; label: string } = $props();
  const percent = $derived(Math.round(Math.min(1, Math.max(0, value)) * 100));
</script>

<div class="download-progress">
  <div class="progress-bar" role="progressbar" aria-label={label} aria-valuemin="0" aria-valuemax="100" aria-valuenow={percent}>
    <div class="progress-fill" style="width: {percent}%"></div>
  </div>
  <span class="progress-pct">{percent}%</span>
</div>

<style>
  .download-progress { display: flex; align-items: center; gap: 8px; flex-shrink: 0; width: 140px; }
  .progress-bar { flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .progress-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.3s ease; }
  .progress-pct { font-size: 10px; color: var(--text-secondary); font-family: var(--font-mono); width: 28px; text-align: right; }
</style>
