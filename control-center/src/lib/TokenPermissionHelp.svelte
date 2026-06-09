<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";

  let open = $state(false);
  let openToRight = $state(true);
  let trigger: HTMLButtonElement;

  function updatePlacement() {
    if (!trigger) return;
    const triggerRect = trigger.getBoundingClientRect();
    const popoverWidth = Math.min(320, window.innerWidth - 48);
    openToRight = window.innerWidth - triggerRect.left >= popoverWidth + 8;
  }

  function toggleHelp() {
    if (!open) updatePlacement();
    open = !open;
  }

  function clickOutside(node: HTMLElement) {
    function handleClick(event: MouseEvent) {
      if (open && event.target instanceof Node && !node.contains(event.target)) {
        open = false;
      }
    }

    document.addEventListener("click", handleClick, true);
    return {
      destroy() {
        document.removeEventListener("click", handleClick, true);
      }
    };
  }

  async function openTokenSettings() {
    open = false;
    try {
      await invoke("open_url", { url: "https://huggingface.co/settings/tokens" });
    } catch (_) {}
  }
</script>

<svelte:window
  onresize={() => {
    if (open) updatePlacement();
  }}
  onkeydown={(event) => {
    if (event.key === "Escape") open = false;
  }}
/>

<span class="help-wrap" use:clickOutside>
  <button
    bind:this={trigger}
    class="help-trigger"
    type="button"
    aria-label="Show required Hugging Face token permission"
    aria-expanded={open}
    title="Required token permission"
    onclick={toggleHelp}
  >
    ?
  </button>

  {#if open}
    <span
      class="help-popover"
      class:open-to-right={openToRight}
      role="dialog"
      aria-label="Required Hugging Face token permission"
    >
      <span class="help-header">
        <strong>Enable gated model access</strong>
        <button
          class="close-btn"
          type="button"
          aria-label="Close token permission help"
          onclick={() => open = false}
        >
          &times;
        </button>
      </span>
      <span class="help-copy">
        For a fine-grained token, check <strong>Read access to contents of all public gated
        repos you can access</strong>.
      </span>
      <video
        src="/hf-gated-token-access.mp4"
        autoplay
        muted
        loop
        playsinline
        preload="metadata"
        aria-label="Selecting the required gated repository permission"
      ></video>
      <button class="settings-btn" type="button" onclick={openTokenSettings}>
        Open token settings
      </button>
    </span>
  {/if}
</span>

<style>
  .help-wrap {
    position: relative;
    display: inline-flex;
    flex: 0 0 auto;
    vertical-align: middle;
  }

  .help-trigger,
  .close-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--border);
    background: var(--bg-panel);
    color: var(--text-primary);
    cursor: pointer;
    line-height: 1;
  }

  .help-trigger {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    font-size: 11px;
    font-weight: 700;
  }

  .help-trigger:hover,
  .help-trigger:focus-visible {
    border-color: var(--accent);
    background: var(--accent);
    color: white;
    outline: none;
  }

  .help-popover {
    position: absolute;
    z-index: 50;
    top: calc(100% + 7px);
    right: 0;
    display: flex;
    width: min(320px, calc(100vw - 48px));
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-secondary);
    box-shadow: 0 10px 28px rgb(0 0 0 / 45%);
    flex-direction: column;
    gap: 8px;
    color: var(--text-primary);
  }

  .help-popover.open-to-right {
    right: auto;
    left: 0;
  }

  .help-header {
    display: flex;
    min-height: 20px;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-size: 11px;
  }

  .close-btn {
    width: 20px;
    height: 20px;
    border-radius: 3px;
    font-size: 16px;
  }

  .close-btn:hover,
  .close-btn:focus-visible {
    border-color: var(--accent);
    color: white;
    outline: none;
  }

  .help-copy {
    color: var(--text-secondary);
    font-size: 10px;
    line-height: 1.4;
  }

  video {
    display: block;
    width: 100%;
    aspect-ratio: 1;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: #090d17;
    object-fit: contain;
  }

  .settings-btn {
    align-self: flex-end;
    padding: 4px 9px;
    border: 1px solid var(--accent);
    border-radius: 3px;
    background: var(--accent);
    color: white;
    cursor: pointer;
    font-size: 10px;
  }

  .settings-btn:hover,
  .settings-btn:focus-visible {
    filter: brightness(1.12);
    outline: none;
  }
</style>
