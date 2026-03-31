<script lang="ts">
  let {
    open,
    rememberChoice,
    busy = false,
    onRememberChange,
    onChoose,
    onCancel,
  }: {
    open: boolean;
    rememberChoice: boolean;
    busy?: boolean;
    onRememberChange: (value: boolean) => void;
    onChoose: (action: "tray" | "quit") => void;
    onCancel: () => void;
  } = $props();
</script>

{#if open}
  <div class="overlay">
    <button type="button" class="backdrop" aria-label="cancel close prompt" onclick={onCancel}></button>
    <div
      class="modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="close-modal-title"
      tabindex="-1"
    >
      <div class="title" id="close-modal-title">close gary4local?</div>
      <div class="body">
        choose whether to quit the application or leave it running in the system tray.
      </div>
      <div class="note">
        leaving it in the tray keeps your local services available.
      </div>
      <label class="remember">
        <input
          type="checkbox"
          checked={rememberChoice}
          disabled={busy}
          onchange={(e) => onRememberChange((e.currentTarget as HTMLInputElement).checked)}
        />
        <span>don't show this again</span>
      </label>
      <div class="actions">
        <button onclick={onCancel} disabled={busy}>cancel</button>
        <button onclick={() => onChoose("tray")} disabled={busy}>leave in tray</button>
        <button class="accent" onclick={() => onChoose("quit")} disabled={busy}>quit app</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .overlay {
    position: fixed;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    z-index: 50;
  }

  .backdrop {
    position: absolute;
    inset: 0;
    border: none;
    background: rgba(0, 0, 0, 0.7);
    padding: 0;
  }

  .modal {
    position: relative;
    z-index: 1;
    width: min(420px, 100%);
    border: 1px solid var(--border);
    background: var(--bg-secondary);
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.45);
    padding: 18px;
  }

  .title {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .body {
    margin-top: 10px;
    font-size: 12px;
    color: var(--text-primary);
    line-height: 1.5;
  }

  .note {
    margin-top: 8px;
    font-size: 11px;
    color: var(--text-secondary);
    line-height: 1.4;
  }

  .remember {
    margin-top: 14px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-primary);
  }

  .remember input {
    accent-color: var(--accent);
  }

  .actions {
    margin-top: 18px;
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }
</style>
