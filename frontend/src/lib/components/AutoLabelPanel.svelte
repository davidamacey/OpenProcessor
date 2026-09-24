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
   *
   * Since 2026-09 the run can optionally be scoped: <AssistScopeBar>
   * picks one class (and, when the backend advertises them, a detection
   * profile and a prompt pack) and contributes `class_id` /
   * `detection_profile` / `prompt_pack` to the start call. The bar is
   * absent entirely unless `/methods` advertises the assist axes, and
   * contributes nothing when the operator leaves it alone — an unscoped
   * "Recluster now" is the same one-click, whole-dataset run it has
   * always been.
   */
  import { onMount, onDestroy } from 'svelte';
  import AssistScopeBar from './AssistScopeBar.svelte';
  import {
    cancelAutoLabel,
    getAutoLabelStatus,
    startAutoLabel,
    unknownStrategyDetail,
    type AutoLabelJobState,
  } from '$lib/api';
  import { createAssistScope } from '$lib/assistScope.svelte';
  import { isScopedAssistAvailable } from '$lib/strategies';
  import { classesStore } from '$stores/classes.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';
  import { toastStore } from '$stores/toast.svelte';

  let job: AutoLabelJobState | null = $state(null);
  let busy: boolean = $state(false);
  let pollTimer: ReturnType<typeof setTimeout> | null = null;
  // Broaden mode: when true the residual AHC stage re-pools items
  // already sitting in candidate clusters so smaller candidates can fuse
  // into bigger ones. Off by default — most runs only want to cluster
  // unassigned + class-bucketed items.
  let mergeCandidates: boolean = $state(false);

  // Cluster scope — train + assign only the largest, clear crops (what the
  // business sorts on); smaller / blurrier crops are parked until a looser
  // recluster. OFF by default so the standard run is unchanged. This is a
  // FULL retrain (not the cheap incremental assign), and it needs the
  // rank/blur backfill complete — the API blocks otherwise.
  // scope: 0 = full pool, 1 = largest only, 2 = largest + 2nd.
  let clusterScope: 0 | 1 | 2 = $state(0);
  let clusterBlur: number = $state(0);
  let nClusters: number | null = $state(null);

  // -- VLM-assist scoping (this plan §4) ---------------------------------
  // The scope bar is absent, not disabled, until /methods advertises the
  // assist axes. `class_id` has no capability signal of its own and an
  // unknown query param is silently dropped server-side, so an ungated
  // picker would start an unscoped hours-long run while claiming it was
  // scoped — see isScopedAssistAvailable's doc comment.
  const scope = createAssistScope();

  // Idempotent, never-rejecting, cached one-shot (degrades to
  // FALLBACK_METHODS on any failure) — same call StrategyBar and /train
  // make. `scopeAvailable` is false for the first frames after mount;
  // that is correct (hide, then reveal) and must not be "fixed" with a
  // spinner or an await.
  $effect(() => {
    void strategiesStore.init();
  });
  const scopeAvailable = $derived(isScopedAssistAvailable(strategiesStore.methods));

  const scopeClassName = $derived(
    scope.classId == null
      ? null
      : (classesStore.byId(scope.classId)?.name ?? `class ${scope.classId}`),
  );

  // This component deliberately does not take its own subscription on
  // classesStore — src/routes/+layout.svelte already holds one for the
  // whole app lifetime, so classesStore.classes is populated and
  // refreshing on every route. A second ref-count here would be
  // redundant with no benefit.

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
    if (seconds < 3600)
      return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
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
        gate_max_rank: clusterScope === 0 ? null : clusterScope,
        gate_min_blur_ratio: clusterBlur > 0 ? clusterBlur : null,
        n_clusters: nClusters && nClusters >= 2 ? nClusters : null,
        // `{}` whenever nothing is scoped, so the composed URL stays
        // byte-identical to every request this panel has ever sent.
        ...scope.toStartParams(),
      });
      toastStore.success(
        scopeClassName
          ? `Recluster started — VLM labeling limited to ${scopeClassName}.`
          : 'Recluster started.',
      );
      schedule();
    } catch (e) {
      const msg = (e as Error).message;
      if (msg.includes('409') || msg.includes('already in progress')) {
        toastStore.warn('A recluster run is already in progress.');
        void poll();
      } else if (unknownStrategyDetail(e)) {
        const d = unknownStrategyDetail(e)!;
        toastStore.error(
          `Start failed: unknown ${d.axis.replace('_', ' ')} "${d.requested}" — valid: ${d.valid_ids.join(', ') || 'none'}.`,
        );
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

  const isRunning: boolean = $derived(
    ((job as AutoLabelJobState | null)?.status ?? '') === 'running',
  );
  const stageIdx: number = $derived(
    stageIndex((job as AutoLabelJobState | null)?.stage ?? ''),
  );
  const percent: number | null = $derived.by(() => {
    if (!job || job.total <= 0) return null;
    return Math.min(100, Math.round((job.processed / job.total) * 100));
  });

  // Backend chip — gpu (cuml …) vs cpu (sklearn …). Empty string when
  // the worker is on an older build that doesn't emit the field, so the
  // chip renders only when the backend has been detected.
  const backendName: string = $derived(
    ((job as AutoLabelJobState | null)?.backend ?? '') as string,
  );
  const backendDetail: string = $derived(
    ((job as AutoLabelJobState | null)?.backend_detail ?? '') as string,
  );
  const peakVramMb: number | null = $derived(
    ((job as AutoLabelJobState | null)?.peak_vram_mb ?? null) as number | null,
  );
  const stageDurations: Array<[string, number]> = $derived.by(() => {
    const sd = (job as AutoLabelJobState | null)?.stage_durations;
    if (!sd) return [];
    return Object.entries(sd) as Array<[string, number]>;
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
        <code class="text-zinc-300">class_id</code>, re-clusters unlabeled residuals
        (AHC), promotes high-purity clusters, and runs Gemma over remaining unvalidated
        crops. v6's confident labels and human validations are never overwritten. Hours at
        350k-crop scale. Safe to cancel.
      </p>
    </div>
    {#if !isRunning}
      <!-- Cluster scope: focus the FULL recluster on the largest, clear
           crops (parks the rest). OFF = current full-pool behavior. Needs
           the rank/blur backfill complete; the API blocks otherwise. -->
      <div class="flex flex-wrap items-center gap-3 text-xs text-zinc-300">
        <span class="text-zinc-500">scope:</span>
        <div class="inline-flex overflow-hidden rounded border border-zinc-700">
          {#each [{ v: 0, l: 'Full pool' }, { v: 1, l: 'Largest' }, { v: 2, l: '+2nd' }] as o (o.v)}
            <button
              type="button"
              class="px-2 py-0.5 {clusterScope === o.v
                ? 'bg-blue-600 text-white'
                : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'}"
              onclick={() => (clusterScope = o.v as 0 | 1 | 2)}
              disabled={busy}
            >
              {o.l}
            </button>
          {/each}
        </div>
        <label
          class="flex items-center gap-1.5"
          title="Train/assign only crops at or above this clarity (blur_lap_ratio)."
        >
          <span class="text-zinc-500">clarity ≥</span>
          <input
            type="range"
            min="0"
            max="2"
            step="0.05"
            bind:value={clusterBlur}
            disabled={busy}
            class="h-1 w-28 cursor-pointer accent-blue-500"
          />
          <span class="w-10 tabular-nums text-zinc-400"
            >{clusterBlur > 0 ? clusterBlur.toFixed(2) : 'off'}</span
          >
        </label>
        <label
          class="flex items-center gap-1.5"
          title="IVF centroid count (default 512). Sweep down with the gate on."
        >
          <span class="text-zinc-500">clusters</span>
          <input
            type="number"
            min="2"
            max="4096"
            placeholder="512"
            bind:value={nClusters}
            disabled={busy}
            class="w-16 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-zinc-100"
          />
        </label>
        {#if clusterScope !== 0 || clusterBlur > 0}
          <span class="text-amber-300/80">full retrain · parks smaller/blurry crops</span>
        {/if}
      </div>
    {/if}
    <div class="flex items-center gap-3">
      {#if !isRunning}
        {#if scopeAvailable}
          <AssistScopeBar {scope} classes={classesStore.classes} disabled={busy} />
        {/if}
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
        <button class="btn btn-danger" type="button" onclick={cancel} disabled={busy}>
          Cancel
        </button>
      {:else}
        <button class="btn btn-primary" type="button" onclick={start} disabled={busy}>
          {scopeClassName ? `Recluster · VLM: ${scopeClassName}` : 'Recluster now'}
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
        <!-- Backend chip: only renders once cluster_residuals has detected
             whether cuML or sklearn ran. Hovering shows the long detail
             string (cuml version + GPU id + free VRAM). -->
        {#if backendName}
          <span
            class="rounded px-2 py-0.5 font-mono uppercase
              {backendName === 'gpu'
              ? 'bg-fuchsia-500/20 text-fuchsia-200'
              : 'bg-zinc-700/40 text-zinc-300'}"
            title={backendDetail}
          >
            {backendName}
          </span>
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
        <div
          class="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px] text-zinc-400"
        >
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

      {#if job.status === 'completed'}
        <!-- Per-stage durations + peak VRAM (GPU runs only). Rendered as
             a compact table above the raw JSON so operators can answer
             "where did the wall time go?" without parsing the full result. -->
        {#if stageDurations.length > 0 || peakVramMb != null || backendDetail}
          <div class="space-y-1 text-xs text-zinc-300">
            {#if backendDetail}
              <p class="font-mono text-[10px] text-zinc-400">{backendDetail}</p>
            {/if}
            {#if stageDurations.length > 0}
              <table class="font-mono text-[10px]">
                <tbody>
                  {#each stageDurations as [name, sec] (name)}
                    <tr>
                      <td class="pr-4 text-zinc-400">{name}</td>
                      <td class="text-zinc-200">{formatDuration(sec)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            {/if}
            {#if peakVramMb != null}
              <p class="font-mono text-[10px] text-zinc-400">
                peak VRAM: {peakVramMb.toLocaleString()} MB
              </p>
            {/if}
          </div>
        {/if}
        {#if job.result?.stages}
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
      {/if}
    </div>
  {/if}
</section>
