<script lang="ts">
  /**
   * Active-run panel: epoch counter, ETA, best-mAP chips, GPU strip.
   * Open MLflow run + Cancel buttons live here too.
   */
  import type { TrainJobStatus } from '$lib/types_train';
  import { externalHref } from '$lib/mlflowLink';
  import { formatDuration } from '$lib/format/duration';
  import { formatMetric } from '$lib/trainResults';

  interface Props {
    status: TrainJobStatus;
    onCancel?: (jobId: string) => void | Promise<void>;
    cancelling?: boolean;
  }

  let { status, onCancel, cancelling = false }: Props = $props();

  const cur = $derived(status.current_epoch ?? 0);
  const tot = $derived(status.total_epochs ?? 0);
  const pct = $derived(tot > 0 ? Math.min(100, Math.round((cur / tot) * 100)) : 0);

  const etaSec = $derived.by(() => {
    if (!status.epoch_time_s || !tot || cur >= tot) return null;
    return Math.max(0, Math.round((tot - cur) * status.epoch_time_s));
  });

  // Live progress has no finished eval pass yet — the last completed
  // TRAINING epoch's own metrics (never best_checkpoint_metric, which
  // Ultralytics only fills once, after training ends) is the honest
  // "how's it doing" number while a run is still running.
  const lastEpoch = $derived(status.last_epoch_metric);

  function statePillClass(s: string): string {
    switch (s) {
      case 'running':
      case 'starting':
        return 'bg-accent/20 text-accent-text border-accent/40';
      case 'exporting':
        return 'bg-holdout/20 text-holdout-text border-holdout/40';
      case 'finished':
        return 'bg-success/20 text-success-text border-success/40';
      case 'failed':
        return 'bg-danger/20 text-danger-text border-danger/40';
      case 'cancelled':
      case 'skipped':
        return 'bg-surface-3 text-fg-2 border-line-strong';
      case 'lost':
        return 'bg-warn/20 text-warn-text border-warn/40';
      case 'queued':
      default:
        return 'bg-surface-2 text-fg-2 border-line-strong';
    }
  }
</script>

<section class="rounded-md border border-line bg-surface-1 p-4">
  <header class="mb-3 flex flex-wrap items-center gap-3">
    <h2 class="font-mono text-body text-white">{status.job_id}</h2>
    <span
      class="rounded-sm border px-1.5 py-0.5 text-caption font-medium capitalize {statePillClass(
        status.state,
      )}"
    >
      {status.state}
    </span>
    <span class="grow"></span>
    {#if externalHref(status.mlflow_run_url)}
      <!-- eslint-disable svelte/no-navigation-without-resolve -- external MLflow server URL, not a SvelteKit route -->
      <a
        href={externalHref(status.mlflow_run_url)}
        target="_blank"
        rel="noopener"
        class="btn"
      >
        Open MLflow run
      </a>
      <!-- eslint-enable svelte/no-navigation-without-resolve -->
    {/if}
    {#if onCancel && (status.state === 'running' || status.state === 'starting' || status.state === 'queued' || status.state === 'exporting')}
      <button
        type="button"
        class="btn btn-danger"
        onclick={() => onCancel?.(status.job_id)}
        disabled={cancelling}
      >
        {cancelling ? 'Cancelling…' : 'Cancel run'}
      </button>
    {/if}
  </header>

  <div class="mb-3 grid grid-cols-1 gap-2 text-body sm:grid-cols-2 lg:grid-cols-4">
    <div>
      <dt class="text-meta text-fg-3">Epoch</dt>
      <dd class="font-mono text-fg-1">
        {cur} / {tot || '—'}
        {#if status.epoch_time_s}
          <span class="text-fg-3">· {status.epoch_time_s.toFixed(1)}s/ep</span>
        {/if}
      </dd>
    </div>
    <div>
      <dt class="text-meta text-fg-3">ETA</dt>
      <dd class="font-mono text-fg-1">{formatDuration(etaSec)}</dd>
    </div>
    <div>
      <dt
        class="text-meta text-fg-3"
        title="The last completed training epoch's own validation pass — not the best checkpoint's, which Ultralytics only re-validates once, after training ends"
      >
        Last epoch mAP50{#if lastEpoch?.epoch != null}
          <span class="normal-case text-fg-3"> (ep {lastEpoch.epoch})</span>
        {/if}
      </dt>
      <dd class="font-mono text-fg-1">{formatMetric(lastEpoch?.map50)}</dd>
    </div>
    <div>
      <dt class="text-meta text-fg-3">Last epoch mAP50-95</dt>
      <dd class="font-mono text-fg-1">{formatMetric(lastEpoch?.map50_95)}</dd>
    </div>
  </div>

  {#if tot > 0}
    <div class="mb-3 h-2 w-full overflow-hidden rounded-full bg-surface-2">
      <div
        class="h-full rounded-full bg-accent transition-all"
        style="width: {pct}%"
      ></div>
    </div>
  {/if}

  {#if status.gpu && status.gpu.length > 0}
    <div class="flex flex-wrap gap-2">
      {#each status.gpu as g, i (g.index ?? i)}
        <span
          class="rounded-md border border-line-strong bg-surface-0 px-2 py-1 font-mono text-meta text-fg-2"
          title="GPU {g.index ?? i} utilization / memory"
        >
          GPU {g.index ?? i}
          <span class="text-fg-1">{g.util_pct ?? '—'}%</span>
          <span class="text-fg-3">·</span>
          <span class="text-fg-1">
            {((g.mem_used_mb ?? 0) / 1024).toFixed(1)} /
            {((g.mem_total_mb ?? 0) / 1024).toFixed(1)} GB
          </span>
        </span>
      {/each}
    </div>
  {/if}

  {#if status.error}
    <p
      class="mt-3 truncate rounded border border-danger/30 bg-danger/10 px-2 py-1 font-mono text-caption text-danger-text"
      title={status.error}
    >
      {status.error}
    </p>
  {/if}
</section>
