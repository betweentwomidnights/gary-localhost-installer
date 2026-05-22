<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { onMount } from "svelte";

  let {
    serviceId = "stable-audio",
    onTokenChange
  }: {
    serviceId?: string;
    onTokenChange: (configured: boolean) => void;
  } = $props();

  let hfTokenConfigured: boolean = $state(false);
  let maskedToken: string = $state("");
  let tokenInput: string = $state("");
  let saving: boolean = $state(false);
  let message: string | null = $state(null);

  const stableAudioOpenUrl = "https://huggingface.co/stabilityai/stable-audio-open-small";
  const sa3MediumUrl = "https://huggingface.co/stabilityai/stable-audio-3-medium";
  const t5GemmaUrl = "https://huggingface.co/google/t5gemma-b-b-ul2";
  let isSa3 = $derived(serviceId === "sa3");

  async function loadToken() {
    try {
      const token = await invoke<string | null>("get_hf_token");
      hfTokenConfigured = !!token;
      if (token) {
        maskedToken = token.slice(0, 6) + "..." + token.slice(-4);
      }
      onTokenChange(hfTokenConfigured);
    } catch (_) {}
  }

  async function saveToken() {
    const t = tokenInput.trim();
    if (!t) return;
    saving = true;
    message = null;
    try {
      await invoke("save_hf_token", { token: t });
      hfTokenConfigured = true;
      maskedToken = t.slice(0, 6) + "..." + t.slice(-4);
      tokenInput = "";
      message = "Token saved!";
      onTokenChange(true);
    } catch (e: any) {
      message = "Failed: " + (typeof e === "string" ? e : e?.message || "unknown");
    } finally {
      saving = false;
    }
  }

  async function removeToken() {
    try {
      await invoke("delete_hf_token");
      hfTokenConfigured = false;
      maskedToken = "";
      message = null;
      onTokenChange(false);
    } catch (_) {}
  }

  async function openUrl(url: string) {
    try {
      await invoke("open_url", { url });
    } catch (_) {}
  }

  onMount(loadToken);
</script>

<div class="token-banner" class:configured={hfTokenConfigured}>
  {#if !hfTokenConfigured}
    <div class="banner-title">huggingface token required for gated audio models</div>
    <div class="steps">
      <div class="step">
        <span class="num">1</span>
        <span>Agree to the model access terms</span>
        {#if isSa3}
          <button class="link-btn" onclick={() => openUrl(sa3MediumUrl)}>SA3 Medium</button>
          <button class="link-btn compact" onclick={() => openUrl(t5GemmaUrl)}>T5Gemma</button>
        {:else}
          <button class="link-btn" onclick={() => openUrl(stableAudioOpenUrl)}>Open model page</button>
        {/if}
      </div>
      <div class="step">
        <span class="num">2</span>
        <span>Create a read token</span>
        <button class="link-btn" onclick={() => openUrl("https://huggingface.co/settings/tokens")}>
          Token settings
        </button>
      </div>
      <div class="step">
        <span class="num">3</span>
        <span>Paste your token:</span>
      </div>
    </div>
    <div class="input-row">
      <input
        type="password"
        bind:value={tokenInput}
        placeholder="hf_..."
        onkeydown={(e) => e.key === "Enter" && saveToken()}
      />
      <button class="save-btn" onclick={saveToken} disabled={saving || !tokenInput.trim()}>
        {saving ? "Saving..." : "Save Token"}
      </button>
    </div>
  {:else}
    <div class="configured-row">
      <span class="banner-title">huggingface token configured</span>
      <span class="masked">{maskedToken}</span>
      <button class="remove-btn" onclick={removeToken}>Remove</button>
    </div>
    {#if isSa3}
      <div class="access-note">
        SA3 also needs accepted access for
        <button class="inline-link" onclick={() => openUrl(sa3MediumUrl)}>SA3 Medium</button>
        and
        <button class="inline-link" onclick={() => openUrl(t5GemmaUrl)}>T5Gemma</button>.
      </div>
    {/if}
  {/if}
  {#if message}
    <div class="msg">{message}</div>
  {/if}
</div>

<style>
  .token-banner {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: #2a1a1a;
  }

  .token-banner.configured {
    background: var(--bg-secondary);
    padding: 8px 16px;
  }

  .banner-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 6px;
  }

  .configured .banner-title {
    margin-bottom: 0;
    color: var(--green);
  }

  .steps {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-bottom: 8px;
  }

  .step {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--accent);
    color: white;
    font-size: 9px;
    font-weight: 700;
    flex-shrink: 0;
  }

  .link-btn {
    font-size: 10px;
    padding: 1px 6px;
    border: 1px solid var(--accent);
    border-radius: 3px;
    background: transparent;
    color: var(--accent);
    cursor: pointer;
    white-space: nowrap;
    margin-left: auto;
  }

  .link-btn.compact {
    margin-left: 0;
  }

  .link-btn:hover {
    background: var(--accent);
    color: white;
  }

  .input-row {
    display: flex;
    gap: 6px;
  }

  .input-row input {
    flex: 1;
    font-size: 11px;
    padding: 4px 8px;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: var(--bg-panel);
    color: var(--text-primary);
    font-family: var(--font-mono);
    outline: none;
  }

  .input-row input:focus {
    border-color: var(--accent);
  }

  .input-row input::placeholder {
    color: var(--text-muted);
  }

  .save-btn {
    font-size: 10px;
    padding: 4px 10px;
    border: none;
    border-radius: 3px;
    background: var(--accent);
    color: white;
    cursor: pointer;
  }

  .save-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .configured-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .masked {
    font-size: 10px;
    font-family: var(--font-mono);
    color: var(--text-muted);
  }

  .remove-btn {
    font-size: 10px;
    padding: 1px 6px;
    border: 1px solid var(--red);
    border-radius: 3px;
    background: transparent;
    color: var(--red);
    cursor: pointer;
    margin-left: auto;
  }

  .remove-btn:hover {
    background: var(--red);
    color: white;
  }

  .msg {
    margin-top: 4px;
    font-size: 10px;
    color: var(--text-secondary);
  }

  .access-note {
    margin-top: 6px;
    font-size: 10px;
    color: var(--text-muted);
  }

  .inline-link {
    border: none;
    background: transparent;
    color: var(--accent);
    font-size: 10px;
    padding: 0 2px;
    cursor: pointer;
  }

  .inline-link:hover {
    text-decoration: underline;
  }
</style>
