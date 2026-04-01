<script lang="ts">
  import { tick } from "svelte";

  let { serviceId, logText }: {
    serviceId: string | null;
    logText: string;
  } = $props();

  let logContainer: HTMLPreElement;
  let autoScroll = $state(true);

  // Auto-scroll when log content changes
  $effect(() => {
    if (logText && logContainer && autoScroll) {
      tick().then(() => {
        logContainer.scrollTop = logContainer.scrollHeight;
      });
    }
  });

  function handleScroll() {
    if (!logContainer) return;
    const { scrollTop, scrollHeight, clientHeight } = logContainer;
    // If user scrolled up more than 50px from bottom, pause auto-scroll
    autoScroll = scrollHeight - scrollTop - clientHeight < 50;
  }
</script>

<div class="log-viewer">
  {#if serviceId}
    <div class="log-header">
      <span class="log-title">Logs: {serviceId}</span>
      {#if !autoScroll}
        <button class="scroll-btn" onclick={() => {
          autoScroll = true;
          if (logContainer) logContainer.scrollTop = logContainer.scrollHeight;
        }}>Scroll to bottom</button>
      {/if}
    </div>
    <pre
      class="log-content"
      bind:this={logContainer}
      onscroll={handleScroll}
    >{logText || "No output yet."}</pre>
  {:else}
    <div class="no-selection">
      <p>Select a service to view logs</p>
    </div>
  {/if}
</div>

<style>
  .log-viewer {
    display: flex;
    flex-direction: column;
    height: 100%;
  }
  .log-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-secondary);
  }
  .log-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
  }
  .scroll-btn {
    font-size: 10px;
    padding: 2px 8px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 3px;
    cursor: pointer;
  }
  .scroll-btn:hover {
    opacity: 0.8;
  }
  .log-content {
    flex: 1;
    overflow-y: auto;
    padding: 12px 16px;
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-all;
    color: var(--text-primary);
    background: var(--bg-primary);
    margin: 0;
    user-select: text;
    -webkit-user-select: text;
    cursor: text;
  }
  .no-selection {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted);
  }
</style>
