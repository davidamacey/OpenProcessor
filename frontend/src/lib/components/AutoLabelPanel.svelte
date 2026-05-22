<script lang="ts">
  /*
   * Recluster / auto-label control panel.
   *
   * Renders on the dashboard. Polls /curation/pipeline/auto_label/status — slowly
   * when idle (30s, just to detect a daemon-fired run), fast when a job is
   * active (3s). Shows stage name, progress bar, ETA, elapsed time, and the
   * final stage-by-stage summary once the run lands.
   *
   * The pipeline (cluster_id_normalize → AHC residuals → auto_promote
   * → gemma → finalize) is the same one the cluster-refresh daemon runs, so
   * kicking it off here is functionally identical to the existing cron
   * path — just with operator visibility. The previous CLIP-prototype
   * labeling stage was removed because a single mean centroid couldn't
   * represent visually diverse classes and produced confident mis-labels
   * that Gemma was then prevented from reviewing.
   */
  import { onMount, onDestroy } from 'svelte';
  import {
    cancelAutoLabel,
    getAutoLabelStatus,
    startAutoLabel,
    type AutoLabelJobState,
  } from '$lib/api';
  import { toastStore } from '$stores/toast.svelte';

  let job: AutoLabelJobState | null = $state(null);
  let busy: boolean = $state(false);
  let pollTimer: ReturnType<typeof setTimeout> | null = null;
  // Broaden mode: when true the residual AHC stage re-pools items
  // already sitting in candidate clusters so smaller candidates can fuse
  // into bigger ones. Off by default — most runs only want to cluster
  // unassigned + class-bucketed items.
  let mergeCandidates: boolean = $state(false);

  // Human-readable label per stage. Order matters — pipeline stages move
  // forward through this list. The percent indicator only renders for
  // 'gemma' because that's the only stage with a meaningful total.
  const STAGE_LABEL: Record<string, string> = {
    '': 'preparing…',
    cluster_id_normalize: 'aligning cluster_id with class_id',
    cluster_residuals: 'clustering residual pool (AHC)',
    auto_promote: 'promoting high-purity clusters',
    gemma: 'Gemma sweep over unvalidated crops',
    finalize: 'final normalization and accounting',
  };

  const STAGE_ORDER = [
    'cluster_id_normalize',
    'cluster_residuals',
    'auto_promote',
    'gemma',
    'finalize',
  ];

  function stageIndex(s: string): number {
    const i = STAGE_ORDER.indexOf(s);
    return i < 0 ? 0 : i;
  }

  function formatDuration(seconds: number | null): string {
    if (seconds == null || !Number.isFinite(seconds)) return '—';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  }

  async function poll(): Promise<void> {
    try {
      job = await getAutoLabelStatus();
    } catch {
      /* network blip — keep last job, try again next tick */
    }
    schedule();
  }

  function schedule(): void {
    if (pollTimer) clearTimeout(pollTimer);
    // Fast poll while running, slow when idle. The slow poll catches
    // daemon-initiated runs without hammering the API.
    const ms = job?.status === 'running' ? 3000 : 30000;
    pollTimer = setTimeout(() => void poll(), ms);
  }

  async function start(): Promise<void> {
    busy = true;
    try {
      job = await startAutoLabel({
        train_clusters: true,
        gemma_concurrency: 16,
        // 0 = process every unvalidated crop. Operators can ramp down
        // later if they want to time-box a run.
        max_gemma_crops: 0,
        recluster_unvalidated: mergeCandidates,
      });
      toastStore.success('Recluster started.');
      schedule();
    } catch (e) {
      const msg = (e as Error).message;
      if (msg.includes('409') || msg.includes('already in progress')) {
        toastStore.warn('A recluster run is already in progress.');
        void poll();
      } else {
        toastStore.error(`Start failed: ${msg}`);
      }
    } finally {
      busy = false;
    }
  }

  async function cancel(): Promise<void> {
    busy = true;
    try {
      job = await cancelAutoLabel();
      toastStore.info('Cancel requested. Pipeline will stop at the next checkpoint.');
    } catch (e) {
      toastStore.error(`Cancel failed: ${(e as Error).message}`);
    } finally {
      busy = false;
    }
  }

  onMount(() => {
    void poll();
  });

  onDestroy(() => {
    if (pollTimer) clearTimeout(pollTimer);
  });

  const isRunning: boolean = $derived(((job as AutoLabelJobState | null)?.status ?? '') === 'running');
  const stageIdx: number = $derived(stageIndex((job as AutoLabelJobState | null)?.stage ?? ''));
  const percent: number | null = $derived.by(() => {
    if (!job || job.total <= 0) return null;
    return Math.min(100, Math.round((job.processed / job.total) * 100));
  });
  const lastFinishedRel: string = $derived.by(() => {
    if (!job || !job.finished_at) return '';
    const seconds = Math.max(0, Date.now() / 1000 - job.finished_at);
    if (seconds < 60) return `${Math.round(seconds)}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  });
</script>

<section class="rounded-md border border-zinc-800 bg-zinc-950 p-4">
  <div class="flex items-center justify-between gap-3">
    <div>
      <h2 class="text-base font-semibold">Data integrity — recluster</h2>
      <p class="mt-0.5 text-xs text-zinc-400">
        Aligns <code class="text-zinc-300">cluster_id</code> with
        <code class="text-zinc-300">class_id</code>, re-clusters
        unlabeled residuals (AHC), promotes high-purity clusters,
        and runs Gemma over remaining unvalidated crops. v6's confident
        labels and human validations are never overwritten. Hours at
        350k-crop scale. Safe to cancel.
      </p>
    </div>
    <div class="flex items-center gap-3">
      {#if !isRunning}
        <label
          class="flex items-center gap-1.5 text-xs text-zinc-300"
          title="Re-pool items already in candidate clusters so smaller candidates can merge into bigger ones."
        >
          <input
            type="checkbox"
            class="h-3.5 w-3.5 accent-blue-500"
            bind:checked={mergeCandidates}
            disabled={busy}
          />
          Merge candidate clusters
        </label>
      {/if}
      {#if isRunning}
        <button
          class="btn btn-danger"
          type="button"
          onclick={cancel}
          disabled={busy}
        >
          Cancel
        </button>
      {:else}
        <button
          class="btn btn-primary"
          type="button"
          onclick={start}
          disabled={busy}
        >
          Recluster now
        </button>
      {/if}
    </div>
  </div>

  {#if job}
    <div class="mt-4 space-y-3">
      <!-- Status chip + stage label. Status drives the color so a glance
           tells the operator whether a previous run failed. -->
      <div class="flex flex-wrap items-center gap-2 text-xs">
        <span
          class="rounded px-2 py-0.5 font-mono uppercase
            {job.status === 'running'
            ? 'bg-blue-500/20 text-blue-200'
            : job.status === 'completed'
              ? 'bg-emerald-500/20 text-emerald-200'
              : job.status === 'failed'
                ? 'bg-red-500/20 text-red-200'
                : job.status === 'cancelled'
                  ? 'bg-orange-500/20 text-orange-200'
                  : 'bg-zinc-700/40 text-zinc-300'}"
        >
          {job.status}
        </span>
        {#if isRunning}
          <span class="text-zinc-300">{STAGE_LABEL[job.stage] ?? job.stage}</span>
          <span class="font-mono text-zinc-500">
            stage {stageIdx + 1}/{STAGE_ORDER.length}
          </span>
        {:else if job.status !== 'idle' && job.finished_at}
          <span class="text-zinc-400">finished {lastFinishedRel}</span>
        {/if}
      </div>

      {#if isRunning}
        <!-- Per-stage progress bar. Indeterminate when total=0 (AHC /
             normalize stages don't expose a counter); determinate for the
             Gemma stage which is the dominant cost. -->
        <div class="h-2 overflow-hidden rounded bg-zinc-800">
          {#if percent != null}
            <div
              class="h-full bg-blue-500 transition-all"
              style="width: {percent}%"
            ></div>
          {:else}
            <div class="h-full w-1/3 animate-pulse bg-blue-500/50"></div>
          {/if}
        </div>
        <div class="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px] text-zinc-400">
          <span>
            {#if job.total > 0}
              {job.processed.toLocaleString()} / {job.total.toLocaleString()}
              {percent != null ? `(${percent}%)` : ''}
            {:else}
              working…
            {/if}
          </span>
          <span>
            elapsed {formatDuration(job.elapsed_seconds)}
            {#if job.eta_seconds != null}
              · ETA {formatDuration(job.eta_seconds)}
            {/if}
          </span>
        </div>
      {/if}

      {#if job.error}
        <p class="text-xs text-red-300">Error: {job.error}</p>
      {/if}

      {#if job.status === 'completed' && job.result?.stages}
        <!-- Once-only summary chip strip. Picks the counts most operators
             care about; the full JSON is available via curl for the rare
             times you need it. -->
        <details class="text-xs text-zinc-300">
          <summary class="cursor-pointer text-zinc-400 hover:text-zinc-200">
            Last run summary
          </summary>
          <pre
            class="mt-2 max-h-64 overflow-auto rounded bg-zinc-900 p-2 font-mono text-[10px] text-zinc-300">{JSON.stringify(
              job.result,
              null,
              2,
            )}</pre>
        </details>
      {/if}
    </div>
  {/if}
</section>
