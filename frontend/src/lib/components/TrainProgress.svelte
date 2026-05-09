<script lang="ts">
  /**
   * Active-run panel: epoch counter, ETA, best-mAP chips, GPU strip.
   * Open MLflow run + Cancel buttons live here too.
   */
  import type { TrainJobStatus } from '$lib/types_train';

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

  function fmtEta(sec: number | null): string {
    if (sec == null) return '—';
    if (sec < 60) return `${sec}s`;
    const m = Math.round(sec / 60);
    if (m < 60) return `${m}m`;
    const h = Math.floor(m / 60);
    const r = m % 60;
    return r === 0 ? `${h}h` : `${h}h ${r}m`;
  }

  function fmtMetric(v: number | undefined): string {
    if (v == null) return '—';
    return v.toFixed(3);
  }

  function statePillClass(s: string): string {
    switch (s) {
      case 'running':
      case 'starting':
        return 'bg-blue-500/20 text-blue-200 border-blue-500/40';
      case 'exporting':
        return 'bg-purple-500/20 text-purple-200 border-purple-500/40';
      case 'finished':
        return 'bg-green-500/20 text-green-200 border-green-500/40';
      case 'failed':
        return 'bg-red-500/20 text-red-200 border-red-500/40';
      case 'cancelled':
      case 'skipped':
        return 'bg-zinc-700 text-zinc-300 border-zinc-600';
      case 'lost':
        return 'bg-yellow-500/20 text-yellow-200 border-yellow-500/40';
      case 'queued':
      default:
        return 'bg-zinc-800 text-zinc-300 border-zinc-700';
    }
  }
</script>

<section class="rounded-md border border-zinc-800 bg-zinc-900 p-4">
  <header class="mb-3 flex flex-wrap items-center gap-3">
    <h2 class="font-mono text-sm text-white">{status.job_id}</h2>
    <span
      class="rounded-sm border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide {statePillClass(
        status.state,
      )}"
    >
      {status.state}
    </span>
    <span class="grow"></span>
    {#if status.mlflow_run_url}
      <a
        href={status.mlflow_run_url}
        target="_blank"
        rel="noopener"
        class="btn"
      >
        Open MLflow run
      </a>
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

  <div class="mb-3 grid grid-cols-1 gap-2 text-sm sm:grid-cols-2 lg:grid-cols-4">
    <div>
      <dt class="text-[11px] uppercase tracking-wide text-zinc-500">Epoch</dt>
      <dd class="font-mono text-zinc-100">
        {cur} / {tot || '—'}
        {#if status.epoch_time_s}
          <span class="text-zinc-500">· {status.epoch_time_s.toFixed(1)}s/ep</span>
        {/if}
      </dd>
    </div>
    <div>
      <dt class="text-[11px] uppercase tracking-wide text-zinc-500">ETA</dt>
      <dd class="font-mono text-zinc-100">{fmtEta(etaSec)}</dd>
    </div>
    <div>
      <dt class="text-[11px] uppercase tracking-wide text-zinc-500">Best mAP50</dt>
      <dd class="font-mono text-zinc-100">{fmtMetric(status.best_metric?.map50)}</dd>
    </div>
    <div>
      <dt class="text-[11px] uppercase tracking-wide text-zinc-500">Best mAP50-95</dt>
      <dd class="font-mono text-zinc-100">{fmtMetric(status.best_metric?.map50_95)}</dd>
    </div>
  </div>

  {#if tot > 0}
    <div class="mb-3 h-2 w-full overflow-hidden rounded-full bg-zinc-800">
      <div
        class="h-full rounded-full bg-blue-500 transition-all"
        style="width: {pct}%"
      ></div>
    </div>
  {/if}

  {#if status.gpu && status.gpu.length > 0}
    <div class="flex flex-wrap gap-2">
      {#each status.gpu as g, i (g.index ?? i)}
        <span
          class="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 font-mono text-xs text-zinc-300"
          title="GPU {g.index ?? i} utilization / memory"
        >
          GPU {g.index ?? i}
          <span class="text-zinc-100">{g.util_pct ?? '—'}%</span>
          <span class="text-zinc-500">·</span>
          <span class="text-zinc-100">
            {((g.mem_used_mb ?? 0) / 1024).toFixed(1)} /
            {((g.mem_total_mb ?? 0) / 1024).toFixed(1)} GB
          </span>
        </span>
      {/each}
    </div>
  {/if}

  {#if status.error}
    <p
      class="mt-3 truncate rounded border border-red-500/30 bg-red-500/10 px-2 py-1 font-mono text-[11px] text-red-200"
      title={status.error}
    >
      {status.error}
    </p>
  {/if}
</section>
