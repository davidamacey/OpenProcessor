<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { getModelsStatus, unloadModel } from '$lib/api';
  import {
    unloadButtonState,
    unloadConfirmMessage,
    unloadForceConfirmMessage,
  } from '$lib/modelUnload';
  import { toastStore } from '$stores/toast.svelte';
  import type { OpModel, OpModelStatus } from '$lib/types';

  const REFRESH_MS = 15_000;

  let models = $state<OpModel[]>([]);
  let loading = $state<boolean>(true);
  let error = $state<string | null>(null);
  let lastUpdated = $state<Date | null>(null);
  let timer: ReturnType<typeof setInterval> | null = null;
  let abortCtrl: AbortController | null = null;
  /** Model name currently mid-unload, or null. Gates the button so a
   *  double-click can't fire two DELETEs for the same model. */
  let unloadingName = $state<string | null>(null);

  async function refresh(): Promise<void> {
    abortCtrl?.abort();
    abortCtrl = new AbortController();
    try {
      const res = await getModelsStatus(abortCtrl.signal);
      models = res.models ?? [];
      lastUpdated = new Date();
      error = null;
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
      error = (e as Error).message;
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    refresh();
    timer = setInterval(refresh, REFRESH_MS);
  });

  onDestroy(() => {
    abortCtrl?.abort();
    if (timer) clearInterval(timer);
  });

  function statusPillClass(s: OpModelStatus): string {
    if (s === 'ready') return 'bg-green-500/20 text-green-200 border-green-500/40';
    if (s === 'not_ready') return 'bg-yellow-500/20 text-yellow-200 border-yellow-500/40';
    return 'bg-red-500/20 text-red-200 border-red-500/40';
  }

  function statusLabel(s: OpModelStatus): string {
    if (s === 'ready') return 'ready';
    if (s === 'not_ready') return 'not ready';
    return 'unavailable';
  }

  function fmtCount(n: number | null): string {
    if (n === null || n === undefined) return '—';
    if (n < 1000) return n.toString();
    if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
    return `${(n / 1_000_000).toFixed(2)}M`;
  }

  function fmtMs(ms: number | null): string {
    if (ms === null || ms === undefined) return '—';
    if (ms < 1) return `${(ms * 1000).toFixed(0)}µs`;
    if (ms < 1000) return `${ms.toFixed(2)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }

  function relTime(d: Date | null): string {
    if (!d) return '';
    const sec = Math.floor((Date.now() - d.getTime()) / 1000);
    if (sec < 5) return 'just now';
    if (sec < 60) return `${sec}s ago`;
    return `${Math.floor(sec / 60)}m ago`;
  }

  /**
   * Unload + delete a model (follow-up gap 2,
   * docs/design/audit-remediation-plan-2026-09.md Appendix D item 3,
   * 2026-09-11). `unloadButtonState` decides what's shown at all — this
   * only handles the click. Force-required models (active vehicle model
   * / other core pipeline models) get a second, stronger confirmation on
   * top of the normal one before ever sending `force=true`; the server
   * is the real guard (LPR models 403 unconditionally) but the double
   * confirm here matches CLAUDE.md's "bulk ops show a confirmation
   * dialog" pattern for a destructive single-model action.
   */
  async function handleUnload(m: OpModel): Promise<void> {
    const state = unloadButtonState(m);
    if (state === 'hidden') return;
    const forced = state === 'force-required';
    if (!window.confirm(unloadConfirmMessage(m))) return;
    if (forced && !window.confirm(unloadForceConfirmMessage(m))) return;

    unloadingName = m.name;
    try {
      const res = await unloadModel(m.name, forced);
      toastStore.success(
        `Unloaded ${res.triton_name}` + (res.warning ? ` — ${res.warning}` : ''),
      );
      await refresh();
    } catch (e) {
      toastStore.error(`Unload failed: ${(e as Error).message}`);
    } finally {
      unloadingName = null;
    }
  }
</script>

<svelte:head>
  <title>Models · legacy Labeler</title>
</svelte:head>

<div class="mx-auto max-w-6xl px-4 py-6">
  <header class="mb-6 flex items-end justify-between">
    <div>
      <h1 class="text-xl font-semibold tracking-tight">Models</h1>
      <p class="mt-1 text-sm text-zinc-400">
        Inference services that drive the legacy labeling pipeline. Triton models live on
        the GPU box; Gemma is an external vLLM service. Auto-refreshes every 15 seconds.
      </p>
    </div>
    <div class="flex items-center gap-3 text-xs text-zinc-500">
      {#if lastUpdated}
        <span>Updated {relTime(lastUpdated)}</span>
      {/if}
      <button
        type="button"
        class="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-300 hover:bg-zinc-800"
        onclick={refresh}
        disabled={loading}
      >
        Refresh
      </button>
    </div>
  </header>

  {#if error}
    <div
      class="mb-4 rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200"
    >
      Failed to load models: {error}
    </div>
  {/if}

  {#if loading && models.length === 0}
    <p class="text-sm text-zinc-500">Loading…</p>
  {:else if models.length === 0}
    <p class="text-sm text-zinc-500">No models reported.</p>
  {:else}
    <ul class="grid grid-cols-1 gap-4 md:grid-cols-2">
      {#each models as m (m.name)}
        <li class="rounded-md border border-zinc-800 bg-zinc-900 p-4">
          <div class="mb-2 flex items-start justify-between gap-3">
            <div class="min-w-0">
              <div class="flex items-center gap-2">
                <h2
                  class="truncate text-base font-semibold text-white"
                  title={m.friendly_name}
                >
                  {m.friendly_name}
                </h2>
                <span
                  class="rounded-sm border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide {statusPillClass(
                    m.status,
                  )}"
                >
                  {statusLabel(m.status)}
                </span>
              </div>
              <p class="mt-1 truncate font-mono text-xs text-zinc-500" title={m.name}>
                {m.name}{m.version ? ` (v${m.version})` : ''}
              </p>
            </div>
            <span
              class="rounded border border-zinc-700 bg-zinc-950 px-2 py-0.5 text-[10px] uppercase tracking-wide text-zinc-400"
              title="Service kind"
            >
              {m.kind}
            </span>
          </div>

          <p class="mb-3 text-sm text-zinc-300">{m.role}</p>

          <dl
            class="grid grid-cols-2 gap-x-4 gap-y-2 border-t border-zinc-800 pt-3 text-sm"
          >
            <div>
              <dt class="text-[11px] uppercase tracking-wide text-zinc-500">Type</dt>
              <dd class="text-zinc-200">{m.model_type}</dd>
            </div>
            <div>
              <dt class="text-[11px] uppercase tracking-wide text-zinc-500">Endpoint</dt>
              <dd
                class="truncate font-mono text-xs text-zinc-300"
                title={m.endpoint ?? ''}
              >
                {m.endpoint ?? '—'}
              </dd>
            </div>
            <div>
              <dt class="text-[11px] uppercase tracking-wide text-zinc-500">
                Inferences
              </dt>
              <dd class="font-mono text-zinc-200">{fmtCount(m.inference_count)}</dd>
            </div>
            <div>
              <dt class="text-[11px] uppercase tracking-wide text-zinc-500">
                Avg latency
              </dt>
              <dd class="font-mono text-zinc-200">{fmtMs(m.avg_latency_ms)}</dd>
            </div>
            {#if m.kind === 'triton'}
              <div>
                <dt class="text-[11px] uppercase tracking-wide text-zinc-500">
                  Batched execs
                </dt>
                <dd class="font-mono text-zinc-200">{fmtCount(m.exec_count)}</dd>
              </div>
              <div>
                <dt class="text-[11px] uppercase tracking-wide text-zinc-500">
                  Failures
                </dt>
                <dd
                  class="font-mono {m.inference_failed && m.inference_failed > 0
                    ? 'text-red-300'
                    : 'text-zinc-200'}"
                >
                  {fmtCount(m.inference_failed)}
                </dd>
              </div>
            {/if}
          </dl>

          {#if m.last_error}
            <p
              class="mt-3 truncate rounded border border-red-500/30 bg-red-500/10 px-2 py-1 font-mono text-[11px] text-red-200"
              title={m.last_error}
            >
              {m.last_error}
            </p>
          {/if}

          {#if m.job_id}
            <p class="mt-3 truncate font-mono text-[11px] text-zinc-500" title={m.job_id}>
              promoted from job {m.job_id}
            </p>
          {/if}

          {#if unloadButtonState(m) !== 'hidden'}
            <div class="mt-3 flex justify-end border-t border-zinc-800 pt-3">
              <button
                type="button"
                class="rounded border px-2 py-1 text-xs transition {unloadButtonState(m) ===
                'force-required'
                  ? 'border-red-700 bg-red-950/40 text-red-200 hover:bg-red-900/40'
                  : 'border-zinc-700 bg-zinc-950 text-zinc-300 hover:border-red-500 hover:text-red-300'}"
                onclick={() => handleUnload(m)}
                disabled={unloadingName === m.name}
                title={unloadButtonState(m) === 'force-required'
                  ? 'Currently serving live traffic — requires a second confirmation'
                  : 'Unload from Triton and delete its model repo directory'}
              >
                {unloadingName === m.name
                  ? 'Unloading…'
                  : unloadButtonState(m) === 'force-required'
                    ? 'Force unload'
                    : 'Unload'}
              </button>
            </div>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</div>
