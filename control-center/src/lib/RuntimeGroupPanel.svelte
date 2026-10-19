<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import NativeBackendPicker from "./NativeBackendPicker.svelte";

  interface BuildStatus { building: boolean; current_step: number; total_steps: number; step_label: string; log: string; error: string | null; }
  interface Member {
    id: string;
    display_name: string;
    status: string;
    runtime: "python" | "native";
    native_backend: string | null;
    build_status: BuildStatus | null;
    has_migration: boolean;
    migration_complete: boolean;
    python_retired: boolean;
  }
  interface RuntimeInfo {
    installed: boolean;
    devOverride: boolean;
    version: string | null;
    manifestVersion: string;
    backend: string | null;
    preference: string;
    fallbackReason: string | null;
    offeredBackends: string[];
    devices: { name: string; backend: string; deviceType: string; memoryTotalBytes: number }[];
  }
  interface Activity { at: number; text: string; failed: boolean; }

  let { label, members, onSelectService, onMigrate }: {
    label: string;
    members: Member[];
    onSelectService: (id: string) => void;
    onMigrate: (id: string) => void;
  } = $props();

  let info = $state<RuntimeInfo | null>(null);
  let activity = $state<Activity[]>([]);
  let busy = $state(false);
  let message = $state<string | null>(null);
  let error = $state<string | null>(null);

  // A member already on the runtime speaks for it; before any is, the first.
  const owner = $derived(members.find((member) => member.runtime === "native")?.id ?? members[0]?.id ?? null);
  const building = $derived(members.find((member) => member.build_status?.building) ?? null);
  const live = (member: Member) => ["running", "starting", "unhealthy"].includes(member.status);
  const stoppable = $derived(members.filter((member) => member.runtime === "native" && live(member)));
  const device = $derived(info?.devices.find((entry) => entry.deviceType !== "cpu" && entry.backend.toLowerCase().includes(info?.backend ?? "")));
  const action = $derived(!info ? null
    : !info.installed ? "install runtime"
    : info.devOverride ? null
    : info.version !== info.manifestVersion ? "update runtime"
    : "reinstall runtime");
  const switching = $derived(!!info?.installed && !info.devOverride && info.preference !== "auto" && info.preference !== info.backend);
  // Reload whenever a member's runtime state changes, not on every progress tick.
  const signature = $derived(members.map((member) =>
    [member.id, member.runtime, member.native_backend, member.migration_complete, member.build_status?.building, member.build_status?.error].join(":")).join("|"));

  async function load() {
    if (!owner) return;
    try {
      [info, activity] = await Promise.all([
        invoke<RuntimeInfo>("get_native_runtime_info", { serviceId: owner }),
        invoke<Activity[]>("get_runtime_activity", { serviceId: owner }),
      ]);
    } catch (cause) { error = String(cause); }
  }
  $effect(() => { void signature; void load(); });

  async function choose(backend: string) {
    if (!owner) return;
    busy = true; message = null; error = null;
    try {
      await invoke("save_app_settings", { settings: { nativeBackends: { [owner]: backend } } });
      await load();
      message = info?.installed ? "Reinstall the runtime to switch backends." : "Install the runtime to use this backend.";
    } catch (cause) { error = String(cause); }
    finally { busy = false; }
  }
  async function install() {
    if (!owner) return;
    busy = true; message = null; error = null;
    try { await invoke("install_shared_runtime", { serviceId: owner }); }
    catch (cause) { error = String(cause); }
    finally { busy = false; }
  }
  function when(at: number) {
    return new Date(at * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }
  function names(list: Member[]) { return list.map((member) => member.display_name).join(", "); }
</script>

<div class="runtime-panel">
  <section class="card">
    <div class="row">
      <div class="copy">
        <div class="title">{label} runtime</div>
        <div class="subtitle">
          {#if !info}checking…
          {:else if info.devOverride}local build on {info.backend}
          {:else if info.installed}{info.version} · {info.backend}{#if device} · {device.name} ({(device.memoryTotalBytes / 1024 ** 3).toFixed(1)} GB){/if}
          {:else}not installed yet{/if}
        </div>
      </div>
      {#if info}<NativeBackendPicker value={info.preference} backends={info.offeredBackends} disabled={busy || !!building} onChange={choose} />{/if}
    </div>
    {#if info?.fallbackReason}<div class="warning">{info.fallbackReason}</div>{/if}
    <div class="actions">
      {#if action}<button class="primary" disabled={busy || !!building || !owner} onclick={install}>{building ? "installing…" : action}</button>{/if}
      <span class="note">
        {#if info && !info.installed}{names(members)} share this runtime; installing it once covers all of them.
        {:else if info && info.version !== info.manifestVersion}{info.manifestVersion} is available.
        {:else if switching}Installed on {info?.backend}, set to {info?.preference}: reinstall to switch.
        {:else if info?.installed}Up to date.{/if}
        {#if stoppable.length && action} Stops {names(stoppable)} first.{/if}
      </span>
    </div>
    {#if message}<div class="note" role="status">{message}</div>{/if}
    {#if error}<div class="error" role="alert">{error}</div>{/if}
  </section>

  <section class="card members" aria-label="services on this runtime">
    {#each members as member (member.id)}
      <div class="member">
        <button class="name" onclick={() => onSelectService(member.id)}>{member.display_name}</button>
        <span class="state" class:on={live(member)}>{member.status}</span>
        <span class="engine">{member.runtime === "native" ? `c++${member.native_backend ? ` · ${member.native_backend}` : ""}` : "python"}</span>
        <span class="step">
          {#if member.has_migration && !member.migration_complete}
            <button disabled={!!building} onclick={() => onMigrate(member.id)}>{member.runtime === "native" ? "finish migration" : member.python_retired ? "set up" : "migrate"}</button>
          {/if}
        </span>
      </div>
    {/each}
  </section>

  <section class="activity" aria-label="runtime activity">
    <div class="group-label">runtime activity</div>
    {#if building?.build_status}
      <div class="line now">{building.display_name}: {building.build_status.step_label}</div>
      <pre class="log">{building.build_status.log.split("\n").slice(-8).join("\n")}</pre>
    {/if}
    {#each [...activity].reverse() as entry (entry.at + entry.text)}
      <div class="line" class:failed={entry.failed}><span class="at">{when(entry.at)}</span>{entry.text}</div>
    {:else}
      {#if !building}<div class="line empty">Installs, updates and each service's switch to C++ appear here.</div>{/if}
    {/each}
  </section>
</div>

<style>
  .runtime-panel { display: flex; flex-direction: column; gap: 10px; padding: 12px 16px; overflow-y: auto; height: 100%; box-sizing: border-box; }
  .card { padding: 10px 12px; border: 1px solid var(--border); border-radius: 4px; background: var(--bg-panel); }
  .row { display: flex; align-items: center; gap: 12px; }
  .copy { flex: 1; min-width: 0; }
  .title { font-size: 12px; font-weight: 600; color: var(--text-primary); }
  .subtitle { margin-top: 3px; font-size: 11px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .actions { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
  .actions button { font-size: 11px; flex-shrink: 0; }
  button.primary { color: var(--accent); border-color: var(--accent); }
  .note { font-size: 10px; color: var(--text-muted); line-height: 1.4; }
  .warning { margin-top: 8px; padding: 6px 8px; font-size: 10px; line-height: 1.4; color: var(--orange); border: 1px solid var(--orange); border-radius: 3px; }
  .error { margin-top: 8px; font-size: 11px; color: var(--red); overflow-wrap: anywhere; }
  .members { padding: 4px 12px; }
  .member { display: grid; grid-template-columns: minmax(0, 1fr) 70px 110px 110px; align-items: center; gap: 8px; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 11px; }
  .member:last-child { border-bottom: none; }
  .member .name { justify-self: start; padding: 0; border: none; background: none; color: var(--text-primary); font-size: 12px; font-weight: 600; cursor: pointer; }
  .member .name:hover { color: var(--accent); }
  .state { color: var(--text-muted); }
  .state.on { color: var(--green); }
  .engine { color: var(--text-secondary); font-family: var(--font-mono); font-size: 10px; }
  .step { justify-self: end; }
  .step button { font-size: 10px; padding: 3px 6px; }
  .group-label { padding: 0 0 6px; font-size: 10px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px; }
  .activity { padding: 0 2px; }
  .line { font-size: 11px; line-height: 1.6; color: var(--text-secondary); font-family: var(--font-mono); overflow-wrap: anywhere; }
  .line.failed { color: var(--red); }
  .line.now { color: var(--text-primary); }
  .line.empty { color: var(--text-muted); font-family: inherit; }
  .at { color: var(--text-muted); margin-right: 8px; }
  .log { margin: 4px 0 8px; padding: 6px 8px; max-height: 140px; overflow: auto; font-size: 10px; line-height: 1.5; color: var(--text-secondary); background: var(--bg-hover); border-radius: 3px; white-space: pre-wrap; }
</style>
