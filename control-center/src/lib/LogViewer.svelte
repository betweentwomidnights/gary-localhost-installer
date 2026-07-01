<script lang="ts">
  import { tick } from "svelte";

  let { serviceId, logText, onLiveUpdatesChange }: {
    serviceId: string | null;
    logText: string;
    onLiveUpdatesChange?: (live: boolean) => void;
  } = $props();

  let logContainer: HTMLPreElement | null = $state(null);
  let visibleLogText = $state("");
  let autoScroll = $state(true);
  let liveUpdates = $state(true);
  let isAutoScrolling = false;
  let isSelecting = false;
  let lastServiceId: string | null = $state(null);
  let lastLogLength = $state(0);
  let selectionText = $state("");
  let mouseDownWasLive = false;
  let hasBufferedUpdates = $derived(logText !== visibleLogText);

  function selectionIsInsideLog() {
    if (!logContainer) return false;
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return false;
    return Boolean(
      selection.anchorNode &&
      selection.focusNode &&
      logContainer.contains(selection.anchorNode) &&
      logContainer.contains(selection.focusNode)
    );
  }

  function updateSelectionText() {
    const selection = window.getSelection();
    selectionText = selectionIsInsideLog() && selection ? selection.toString() : "";
  }

  function setLiveUpdates(value: boolean) {
    if (liveUpdates === value) return;
    liveUpdates = value;
    onLiveUpdatesChange?.(value);
  }

  function pauseLiveUpdates() {
    setLiveUpdates(false);
    autoScroll = false;
  }

  async function scrollToBottom() {
    if (!logContainer) return;
    isAutoScrolling = true;
    await tick();
    if (logContainer) {
      logContainer.scrollTop = logContainer.scrollHeight;
    }
    requestAnimationFrame(() => { isAutoScrolling = false; });
  }

  async function resumeLiveUpdates() {
    setLiveUpdates(true);
    autoScroll = true;
    visibleLogText = logText;
    lastLogLength = logText.length;
    await scrollToBottom();
  }

  function handleMouseDown() {
    mouseDownWasLive = liveUpdates && autoScroll;
    isSelecting = true;
    pauseLiveUpdates();
  }

  function handleMouseUp() {
    setTimeout(() => {
      isSelecting = false;
      updateSelectionText();
      if (mouseDownWasLive && !selectionIsInsideLog()) {
        void resumeLiveUpdates();
      }
    }, 100);
  }

  async function copySelection() {
    updateSelectionText();
    if (!selectionText) return;
    await navigator.clipboard.writeText(selectionText);
  }

  $effect(() => {
    if (serviceId !== lastServiceId) {
      lastServiceId = serviceId;
      setLiveUpdates(true);
      autoScroll = true;
      visibleLogText = logText;
      selectionText = "";
      lastLogLength = logText.length;
    }
  });

  $effect(() => {
    const currentLength = logText.length;
    if (liveUpdates) {
      visibleLogText = logText;
    }
    if (liveUpdates && currentLength < lastLogLength) {
      autoScroll = true;
    }
    if (liveUpdates) {
      lastLogLength = currentLength;
    }
  });

  $effect(() => {
    const onSelectionChange = () => updateSelectionText();
    document.addEventListener("selectionchange", onSelectionChange);
    return () => document.removeEventListener("selectionchange", onSelectionChange);
  });

  $effect(() => {
    if (visibleLogText && logContainer && autoScroll && liveUpdates) {
      if (isSelecting || selectionIsInsideLog()) return;
      void scrollToBottom();
    }
  });

  function handleScroll() {
    if (!logContainer || isAutoScrolling) return;
    const { scrollTop, scrollHeight, clientHeight } = logContainer;
    const nearBottom = scrollHeight - scrollTop - clientHeight < 50;
    if (!nearBottom) {
      pauseLiveUpdates();
    }
  }

  function selectLogText() {
    if (!logContainer) return;
    pauseLiveUpdates();
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(logContainer);
    selection?.removeAllRanges();
    selection?.addRange(range);
    updateSelectionText();
    logContainer.focus();
  }
</script>

<div class="log-viewer">
  {#if serviceId}
    <div class="log-header">
      <span class="log-title">Logs: {serviceId}</span>
      <div class="log-actions">
        {#if !liveUpdates}
          <span class="log-state" class:has-updates={hasBufferedUpdates}>
            {hasBufferedUpdates ? "paused + new output" : "paused"}
          </span>
        {/if}
        {#if liveUpdates}
          <button type="button" class="scroll-btn" onclick={pauseLiveUpdates}>pause</button>
        {:else}
          <button type="button" class="scroll-btn accent" onclick={resumeLiveUpdates}>resume live</button>
        {/if}
        {#if selectionText}
          <button
            type="button"
            class="scroll-btn"
            onmousedown={(event) => event.preventDefault()}
            onclick={copySelection}
          >
            copy selection
          </button>
        {/if}
        <button type="button" class="scroll-btn" onclick={selectLogText}>select all</button>
      </div>
    </div>
    <div
      class="log-content-wrap"
      role="presentation"
      onmousedown={handleMouseDown}
      onmouseup={handleMouseUp}
    >
      <pre
        class="log-content"
        bind:this={logContainer}
        tabindex="-1"
        onscroll={handleScroll}
      >{visibleLogText || "No output yet."}</pre>
    </div>
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
    flex: 1;
    min-height: 0;
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
  .scroll-btn.accent {
    background: var(--accent);
  }
  .log-actions {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .log-state {
    font-size: 10px;
    color: var(--text-muted);
  }
  .log-state.has-updates {
    color: var(--accent);
  }
  .log-content {
    display: block;
    flex: 1;
    min-height: 0;
    width: 100%;
    box-sizing: border-box;
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
    outline: none;
  }
  .log-content-wrap {
    display: flex;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }
  .no-selection {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted);
  }
</style>
