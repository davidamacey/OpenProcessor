<script lang="ts">
  /*
   * Pipeline-stats dashboard panel.
   *
   * Subscribes to `GET {API_PREFIX}/pipeline/events` via Server-Sent Events. The
   * server pushes:
   *   * `snapshot` on connect — initial state + dataset stats
   *   * `stats` whenever the dataset rollup actually changed (stage
   *     boundary, terminal status). NOT a fixed-rate stream.
   *   * `state` for pipeline progress (we ignore it here; consumed by
   *     the pipeline-status panel).
   * Replaces the 10s-poll loop from 2026-05-21 — the dashboard now
   * updates on push without burning CPU when nothing is happening on
   * the pipeline.
   *
   * Renders:
   *   - Total crops (large headline)
   *   - Labeled-by-source table with proportion bars
   *   - Unlabeled breakdown (pending_detection / pending_verification /
   *     no_label_source)
   *   - In-progress queue (region_drain_total_unfinished)
   *   - Last clustering run summary (timestamp, method, cluster_count,
   *     residual_count, noise_count)
   */
  import { onDestroy } from 'svelte';
  import { type DatasetStats } from '$lib/api';
  import { subscribePipelineEvents, type CurationEventSubscription } from '$lib/sse';
  import { registeredSlots } from '$lib/annotations/registeredSlots';
  import { resolveStatsUpdate, summarizeStatsError } from '$lib/datasetStats';

  // The slot whose `stats` capability titles this panel (today:
  // license_plate's 'plates'/'Plate detections'/'Plate coverage', all
  // already verbatim in licensePlate.ts). No slot with a stats
  // capability -> no panel at all, rather than an empty "Plate
  // detections" card for a slot-less deployment.
  const statsSlot = $derived(registeredSlots.find((s) => s.stats));

  // Polling interval prop preserved for back-compat with existing
  // callers; ignored now that we're push-driven.
  interface Props {
    /** @deprecated polling removed 2026-05-21 — kept so callers don't break. */
    pollMs?: number;
  }
  // Destructured, not a bare `$props()` binding: Svelte can't infer the
  // exposed property set from a non-destructured declaration.
  let { pollMs: _pollMs }: Props = $props();

  let stats = $state<DatasetStats | null>(null);
  let error = $state<string | null>(null);
  let lastUpdated = $state<number | null>(null);
  let connected = $state<boolean>(false);
  let subscription: CurationEventSubscription | null = null;

  // Drain-rate samples for ETA. We keep a small ring buffer of
  // (unfinished, t) samples so the displayed rate is averaged over the
  // last ~minute instead of a single tick — kills the jitter when
  // sam-worker bursts on a chunk of Gemma-visible-filter results.
  type Sample = { unfinished: number; t: number };
  let samples = $state<Sample[]>([]);
  const SAMPLE_WINDOW_MS = 60_000;

  function applyStats(payload: Record<string, unknown>): void {
    // The backend's {API_PREFIX}/stats/dataset response is usually
    // DatasetStats-shaped, but can be an `{error}` envelope (G1 — see
    // resolveStatsUpdate's doc comment). Never assign that blindly.
    const result = resolveStatsUpdate(payload, stats);
    stats = result.stats;
    error = result.error;
    if (result.error) return;
    lastUpdated = Date.now();
    const unfinished = stats?.in_progress?.region_drain_total_unfinished ?? 0;
    const now = Date.now();
    samples = [...samples, { unfinished, t: now }].filter(
      (s) => now - s.t <= SAMPLE_WINDOW_MS * 2,
    );
  }

  $effect(() => {
    subscription = subscribePipelineEvents({
      onSnapshot: (_state, statsPayload) => {
        connected = true;
        applyStats(statsPayload);
      },
      onStats: (statsPayload) => {
        connected = true;
        applyStats(statsPayload);
      },
      onError: () => {
        connected = false;
        // Don't clear stats — keep last-known values on the screen
        // while the browser/helper reconnects.
      },
      onOpen: () => {
        connected = true;
      },
    });
    return () => {
      subscription?.close();
      subscription = null;
    };
  });

  onDestroy(() => {
    subscription?.close();
    subscription = null;
  });

  const labeledTotal = $derived.by(() => {
    const l = stats?.labeled;
    if (!l) return 0;
    return l.by_human + l.by_vlm + l.by_classifier + l.by_proposal + l.other;
  });

  const labeledRows = $derived.by(() => {
    const l = stats?.labeled;
    if (!l || labeledTotal === 0) return [];
    const rows: Array<{
      key: string;
      label: string;
      count: number;
      tone: string;
    }> = [
      {
        key: 'by_human',
        label: 'Human',
        count: l.by_human,
        tone: 'bg-green-500',
      },
      {
        key: 'by_vlm',
        label: 'VLM',
        count: l.by_vlm,
        tone: 'bg-blue-500',
      },
      {
        key: 'by_classifier',
        label: 'Classifier',
        count: l.by_classifier,
        tone: 'bg-purple-500',
      },
      {
        key: 'by_proposal',
        label: 'Proposal (unclassified)',
        count: l.by_proposal,
        tone: 'bg-amber-500',
      },
      { key: 'other', label: 'Other', count: l.other, tone: 'bg-zinc-500' },
    ];
    return rows.map((r) => ({
      ...r,
      pct: Math.round((r.count / labeledTotal) * 1000) / 10,
    }));
  });

  // Region-detection breakdown — separate from class labeling. The
  // backend's `regions` block counts by role in the active detection
  // profile (detector, segmenter fallback, human), so these rows are
  // generic. Denominator = total_crops (so % is "fraction of crops where
  // a region was detected"), not labeledTotal.
  const slotRows = $derived.by(() => {
    const p = statsSlot?.stats
      ? (stats as unknown as Record<string, DatasetStats['regions'] | undefined>)?.[
          statsSlot.stats.key
        ]
      : undefined;
    const total = stats?.total_crops ?? 0;
    if (!p || total === 0) return [];
    const rows: Array<{
      key: string;
      label: string;
      count: number;
      tone: string;
    }> = [
      {
        key: 'by_detector',
        label: 'Detector',
        count: p.by_detector,
        tone: 'bg-cyan-500',
      },
      {
        key: 'by_segmenter',
        label: 'Segmenter fallback',
        count: p.by_segmenter,
        tone: 'bg-teal-500',
      },
      {
        key: 'by_human_drew',
        label: 'Human-drew bbox',
        count: p.by_human_drew ?? p.by_human,
        tone: 'bg-green-500',
      },
      // Plates the operator confirmed (AI proposed the bbox, human
      // hit Confirm). Distinct from by_human_drew (only when the
      // operator created the bbox from scratch). validated_by_human is
      // the union — every plate the operator touched.
      {
        key: 'validated_by_human',
        label: 'Confirmed by human',
        count: p.validated_by_human ?? 0,
        tone: 'bg-emerald-500',
      },
    ];
    return rows.map((r) => ({
      ...r,
      pct: Math.round((r.count / total) * 1000) / 10,
    }));
  });

  const unlabeledTotal = $derived.by(() => {
    const u = stats?.unlabeled;
    if (!u) return 0;
    return u.pending_detection + u.pending_verification + u.no_label_source;
  });

  // Drain rate (crops/sec) over the last ~minute. Returns null until we
  // have at least 2 samples spanning ≥10s; returns 0 for "queue is empty
  // or not draining at all"; returns a positive number when work is
  // being processed (newest_unfinished < oldest_unfinished).
  const drainRate = $derived.by<number | null>(() => {
    if (samples.length < 2) return null;
    const oldest = samples[0];
    const newest = samples[samples.length - 1];
    const dt = (newest.t - oldest.t) / 1000;
    if (dt < 10) return null;
    // Positive rate = queue draining. Negative = queue growing
    // (ingest faster than worker). We clamp at 0 for ETA math but
    // surface the negative case to the operator.
    return (oldest.unfinished - newest.unfinished) / dt;
  });

  // ETA = remaining_crops / drain_rate. Returns null when we can't yet
  // compute a rate (not enough samples) or the rate is non-positive
  // (queue not shrinking). Operator sees "computing…" or "queue stalled".
  const etaSeconds = $derived.by<number | null>(() => {
    if (drainRate == null || drainRate <= 0) return null;
    const remaining = stats?.in_progress?.region_drain_total_unfinished ?? 0;
    return remaining / drainRate;
  });

  function formatDuration(seconds: number): string {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
    if (seconds < 86400) {
      const h = Math.floor(seconds / 3600);
      const m = Math.round((seconds % 3600) / 60);
      return m === 0 ? `${h}h` : `${h}h ${m}m`;
    }
    const d = Math.floor(seconds / 86400);
    const h = Math.round((seconds % 86400) / 3600);
    return h === 0 ? `${d}d` : `${d}d ${h}h`;
  }

  function formatRate(rate: number): string {
    if (rate >= 10) return `${rate.toFixed(0)} crops/s`;
    if (rate >= 1) return `${rate.toFixed(1)} crops/s`;
    return `${(rate * 60).toFixed(1)} crops/min`;
  }

  function fmt(n: number | undefined | null): string {
    return (n ?? 0).toLocaleString();
  }

  function fmtTimestamp(iso: string | null): string {
    if (!iso) return '—';
    try {
      return new Date(iso).toLocaleString();
    } catch {
      return iso;
    }
  }

  function fmtRelative(ms: number | null): string {
    if (ms == null) return '—';
    const secs = Math.max(0, Math.round((Date.now() - ms) / 1000));
    if (secs < 60) return `${secs}s ago`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
    return `${Math.floor(secs / 3600)}h ago`;
  }
</script>

<section class="space-y-4">
  <header class="flex items-center justify-between">
    <h2 class="text-sm font-semibold text-zinc-300">Pipeline stats</h2>
    <div class="flex items-center gap-2 text-xs text-zinc-500">
      <span title={lastUpdated ? new Date(lastUpdated).toLocaleString() : ''}>
        updated {fmtRelative(lastUpdated)}
      </span>
      <!-- m29 (2026-09-24 interactive pass): this badge tracked only the
           SSE transport (`connected`), so it kept reading "LIVE" in
           green even while the stats banner below said the last frame
           was an error envelope — a transport that's connected but
           streaming errors isn't "live" in any sense an operator cares
           about. -->
      <span
        class="rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wide"
        class:bg-emerald-500={connected && !error}
        class:text-emerald-50={connected && !error}
        class:bg-amber-600={connected && !!error}
        class:text-amber-50={connected && !!error}
        class:bg-zinc-700={!connected}
        class:text-zinc-300={!connected}
        title={!connected
          ? 'Reconnecting…'
          : error
            ? 'Connected, but the last stats frame was an error — showing last known values'
            : 'Live via the pipeline/events SSE stream'}
      >
        {!connected ? '…' : error ? 'degraded' : 'live'}
      </span>
    </div>
  </header>

  {#if error}
    <div
      class="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200"
      title={error}
    >
      Stats unavailable: {summarizeStatsError(error)}
      {#if stats}(showing the last known values){/if}
    </div>
  {/if}
  {#if !stats}
    {#if !error}
      <p class="text-sm text-zinc-500">Loading…</p>
    {/if}
  {:else}
    <div class="grid grid-cols-1 gap-4 lg:grid-cols-4">
      <!-- Headline -->
      <div class="surface p-4 lg:col-span-1">
        <div class="text-xs uppercase tracking-wide text-zinc-500">Total crops</div>
        <div class="mt-2 font-mono text-3xl font-semibold text-zinc-100">
          {fmt(stats.total_crops)}
        </div>
        <dl class="mt-4 space-y-1.5 text-xs">
          <div class="flex justify-between">
            <dt class="text-zinc-400">Validated (class)</dt>
            <dd class="font-mono text-green-300">{fmt(stats.validated)}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-zinc-400">Test holdout</dt>
            <dd class="font-mono">{fmt(stats.test_holdout)}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-zinc-400">Distinct HDD sources</dt>
            <dd class="font-mono">{stats.by_source.length}</dd>
          </div>
        </dl>
      </div>

      <!-- Labeled breakdown -->
      <div class="surface p-4 lg:col-span-2">
        <header class="mb-3 flex items-center justify-between">
          <h3 class="text-sm font-semibold text-zinc-300">Labeled by source</h3>
          <span class="text-xs text-zinc-500">{fmt(labeledTotal)} total</span>
        </header>
        {#if labeledRows.length === 0}
          <p class="text-sm text-zinc-500">No labels yet.</p>
        {:else}
          <ul class="space-y-1.5">
            {#each labeledRows as row (row.key)}
              <li class="flex items-center gap-3 text-xs">
                <span class="w-24 shrink-0 text-zinc-300">{row.label}</span>
                <div class="relative h-3 grow overflow-hidden rounded bg-zinc-900">
                  <div
                    class="h-full {row.tone}"
                    style:width="{Math.max(0.5, row.pct)}%"
                    title="{row.pct}%"
                  ></div>
                </div>
                <span class="w-20 shrink-0 text-right font-mono text-zinc-300">
                  {fmt(row.count)}
                </span>
                <span class="w-12 shrink-0 text-right font-mono text-zinc-500">
                  {row.pct.toFixed(1)}%
                </span>
              </li>
            {/each}
          </ul>
        {/if}
      </div>

      <!-- Last clustering -->
      <div class="surface p-4 lg:col-span-1">
        <h3 class="mb-3 text-sm font-semibold text-zinc-300">Last clustering</h3>
        <dl class="space-y-1.5 text-xs">
          <div class="flex justify-between">
            <dt class="text-zinc-400">When</dt>
            <dd class="text-zinc-300" title={stats.clusters.last_run_at ?? '—'}>
              {fmtTimestamp(stats.clusters.last_run_at)}
            </dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-zinc-400">Clusters (total now)</dt>
            <dd class="font-mono" data-testid="dataset-cluster-count">
              {fmt(stats.clusters.cluster_count)}
            </dd>
          </div>
          {#if stats.clusters.last_run_cluster_count != null}
            <div class="flex justify-between">
              <dt class="text-zinc-400">Made by last run</dt>
              <dd class="font-mono">{fmt(stats.clusters.last_run_cluster_count)}</dd>
            </div>
          {/if}
          <div class="flex justify-between">
            <dt class="text-zinc-400">Residual</dt>
            <dd class="font-mono">{fmt(stats.clusters.residual_count)}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-zinc-400">Noise (cid&lt;0)</dt>
            <dd class="font-mono">{fmt(stats.clusters.noise_count)}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-zinc-400">Method</dt>
            <dd class="font-mono text-zinc-300">
              {stats.clusters.method ?? '—'}
            </dd>
          </div>
        </dl>
      </div>
    </div>

    <!-- Slot detections — separate from class labels. Titled from the
         active stats-capable slot's own spec (today: license_plate's
         'Plate detections'/'Plate coverage'). LPR runs on every crop and
         tries to find a plate bbox; SAM3 is the fallback for when LPR
         misses; humans place plates via the labeler UI. The denominator
         is total_crops, so % = "fraction of crops with a detection". No
         stats-capable slot registered -> no panel at all (see
         statsSlot above). -->
    {#if statsSlot?.stats}
      {@const statsSpec = statsSlot.stats}
      {@const p = (
        stats as unknown as Record<string, DatasetStats['regions'] | undefined>
      )[statsSpec.key]}
      <div class="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div class="surface p-4 lg:col-span-2">
          <header class="mb-3 flex items-center justify-between">
            <h3 class="text-sm font-semibold text-zinc-300">{statsSpec.panelTitle}</h3>
            <span class="text-xs text-zinc-500">
              {fmt(p?.boxed ?? 0)} with a box ·
              {fmt(p?.confirmed ?? 0)} confirmed
            </span>
          </header>
          {#if slotRows.length === 0}
            <p class="text-sm text-zinc-500">No detections yet.</p>
          {:else}
            <ul class="space-y-1.5">
              {#each slotRows as row (row.key)}
                <li class="flex items-center gap-3 text-xs">
                  <span class="w-32 shrink-0 text-zinc-300">{row.label}</span>
                  <div class="relative h-3 grow overflow-hidden rounded bg-zinc-900">
                    <div
                      class="h-full {row.tone}"
                      style:width="{Math.max(0.5, row.pct)}%"
                      title="{row.pct}% of total crops"
                    ></div>
                  </div>
                  <span class="w-20 shrink-0 text-right font-mono text-zinc-300">
                    {fmt(row.count)}
                  </span>
                  <span class="w-12 shrink-0 text-right font-mono text-zinc-500">
                    {row.pct.toFixed(1)}%
                  </span>
                </li>
              {/each}
            </ul>
          {/if}
        </div>

        <!-- Coverage summary -->
        <div class="surface p-4">
          <h3 class="mb-3 text-sm font-semibold text-zinc-300">
            {statsSpec.coverageTitle}
          </h3>
          <div class="flex items-baseline gap-2">
            <span class="font-mono text-2xl text-zinc-100">
              {(((p?.boxed ?? 0) / Math.max(1, stats.total_crops)) * 100).toFixed(1)}%
            </span>
            <span class="text-xs text-zinc-500">of crops have a box</span>
          </div>
          <!-- DQ-m11 (docs/design/data-quality-pass-2026-09-24.md): this
               used to assert "(includes rejected/failed attempts)" about
               `total_detected` as fact. The field's own contract
               (DatasetStats.regions.total_detected in api.ts) says it
               should be a superset of `boxed` — but live, it equals
               `boxed` exactly (1,000 both), so 741 verify-rejected and
               204 no-box crops that DID go through a detector aren't
               reflected in it. Rather than assert something the served
               number doesn't actually demonstrate, state what `boxed` is
               (the honest, verified count) and flag total_detected as a
               credit tally that may not include every rejected/no-box
               attempt on this deployment — not a confident inclusion
               claim. -->
          <p class="mt-2 text-xs text-zinc-500">
            {fmt(p?.boxed ?? 0)} crops carry a box ({fmt(p?.confirmed ?? 0)} Gemma-confirmed).
            The remaining
            {fmt(stats.total_crops - (p?.boxed ?? 0))} either had no visible detection (Gemma
            pre-filter said no), were rejected after detection, or the LPR/SAM3 detectors haven't
            reached them yet.
            {fmt(p?.total_detected ?? 0)} crops carry detector credit (<code
              class="font-mono">regions.total_detected</code
            >) — this may not cover every rejected or box-less attempt.
          </p>
        </div>
      </div>
    {/if}

    <div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <!-- Unlabeled / pending -->
      <div class="surface p-4">
        <header class="mb-3 flex items-center justify-between">
          <h3 class="text-sm font-semibold text-zinc-300">Unlabeled</h3>
          <span class="text-xs text-zinc-500">{fmt(unlabeledTotal)} crops</span>
        </header>
        <dl class="space-y-1.5 text-sm">
          <div class="flex justify-between">
            <dt class="text-zinc-400">Pending detection (SAM)</dt>
            <dd class="font-mono text-orange-300">
              {fmt(stats.unlabeled.pending_detection)}
            </dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-zinc-400">Pending verification (Gemma)</dt>
            <dd class="font-mono text-orange-300">
              {fmt(stats.unlabeled.pending_verification)}
            </dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-zinc-400">No label source (class_id missing)</dt>
            <dd class="font-mono text-red-300">
              {fmt(stats.unlabeled.no_label_source)}
            </dd>
          </div>
        </dl>
      </div>

      <!-- In-progress queue + ETA -->
      <div class="surface p-4">
        <h3 class="mb-3 text-sm font-semibold text-zinc-300">In-flight pipeline</h3>
        <div class="flex items-baseline gap-3">
          <span class="font-mono text-2xl text-zinc-100">
            {fmt(stats.in_progress.region_drain_total_unfinished)}
          </span>
          <span class="text-sm text-zinc-400"
            >crops awaiting detection / verification</span
          >
        </div>

        {#if stats.in_progress.region_drain_total_unfinished > 0}
          <!-- ETA — rolling drain rate over last ~60s. Caveat: only as
               good as the steady-state assumption (sam-worker bursts
               look like big rate spikes; queue refills look like
               regressions). Surface the raw rate alongside the ETA so
               the operator can sanity-check. -->
          <dl class="mt-3 space-y-1 text-xs">
            <div class="flex items-baseline justify-between">
              <dt class="text-zinc-400">ETA to drain</dt>
              <dd class="font-mono text-zinc-100">
                {#if etaSeconds == null}
                  <span class="text-zinc-500" title="Need ≥10s + 2 samples to compute">
                    {drainRate == null
                      ? 'computing…'
                      : drainRate <= 0
                        ? 'queue not shrinking'
                        : '—'}
                  </span>
                {:else}
                  {formatDuration(etaSeconds)}
                {/if}
              </dd>
            </div>
            <div class="flex items-baseline justify-between">
              <dt class="text-zinc-400">Drain rate</dt>
              <dd class="font-mono">
                {#if drainRate == null}
                  <span class="text-zinc-500">…</span>
                {:else if drainRate < 0}
                  <span class="text-red-300">+{formatRate(-drainRate)} (growing)</span>
                {:else if drainRate === 0}
                  <span class="text-zinc-500">0 (stalled)</span>
                {:else}
                  <span class="text-blue-300">{formatRate(drainRate)}</span>
                {/if}
              </dd>
            </div>
          </dl>

          <!-- Indeterminate "the queue is draining" bar -->
          <div class="mt-3 h-1.5 overflow-hidden rounded bg-zinc-900">
            <div class="h-full w-1/3 animate-pulse rounded bg-blue-500"></div>
          </div>
        {:else}
          <p class="mt-3 text-xs text-green-400">queue drained</p>
        {/if}
        <p class="mt-3 text-xs text-zinc-500">
          Matches the API's <code class="font-mono">ingest/region_drain</code> total. While
          &gt; 0 the ingest walker waits before triggering the next clustering pass. SAM3 only
          runs on crops that pass the Gemma visible-filter — most time is spent in Gemma, not
          SAM3.
        </p>
      </div>
    </div>
  {/if}
</section>
