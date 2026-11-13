<script lang="ts">
  import { untrack } from 'svelte';
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
   *     no_label_source / vlm_no_class / by_proposal)
   *   - In-progress queue (region_drain_total_unfinished)
   *   - Last clustering run summary (timestamp, method, cluster_count,
   *     residual_count, noise_count)
   */
  import { formatDuration } from '$lib/format/duration';
  import type { DatasetStats } from '$lib/api_stats';
  import { registeredSlots } from '$lib/annotations/registeredSlots';
  import { embeddingStateChips } from '$components/embedding/embeddingCopy';
  import { createDatasetStatsFeed } from '$lib/datasetStatsFeed.svelte';
  import {
    fmt,
    fmtRelative,
    formatRate,
    labeledRowsOf,
    slotRowsOf,
    summarizeStatsError,
  } from '$lib/datasetStats';
  import { formatTimestamp } from '$lib/format/date';

  // The slot whose `stats` capability titles this panel (its stats key,
  // panel title and coverage title). No slot with a stats capability ->
  // no panel at all, rather than an empty detections card for a
  // slot-less deployment.
  const statsSlot = $derived(registeredSlots.find((s) => s.stats));

  type StatsFeed = ReturnType<typeof createDatasetStatsFeed>;
  interface Props {
    /** The page's feed, when other panels read the same stats; one SSE connection is enough. */
    feed?: StatsFeed;
  }
  const { feed: shared }: Props = $props();

  // Created once at init: a prop that arrives later would not change which stream is live.
  const feed = untrack(() => shared) ?? createDatasetStatsFeed();
  const stats = $derived(feed.stats);
  const error = $derived(feed.error);
  const lastUpdated = $derived(feed.lastUpdated);
  const connected = $derived(feed.connected);
  const drainRate = $derived(feed.drainRate);
  const etaSeconds = $derived(feed.etaSeconds);

  const stateChips = $derived(embeddingStateChips(stats?.embedding?.by_state));
  const labeledTotal = $derived(stats?.labeled?.total ?? 0);
  const labeledRows = $derived(labeledRowsOf(stats));
  const slotRows = $derived(slotRowsOf(stats, statsSlot?.stats?.key));
</script>

<section class="space-y-4">
  <header class="flex items-center justify-between">
    <h2 class="text-body font-semibold text-fg-2">Pipeline stats</h2>
    <div class="flex items-center gap-2 text-meta text-fg-3">
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
        class="rounded px-1.5 py-0.5 text-meta font-semibold"
        class:bg-success={connected && !error}
        class:text-success-text={connected && !error}
        class:bg-warn={connected && !!error}
        class:text-warn-text={connected && !!error}
        class:bg-surface-3={!connected}
        class:text-fg-2={!connected}
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
      class="rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-body text-danger-text"
      title={error}
    >
      Stats unavailable: {summarizeStatsError(error)}
      {#if stats}(showing the last known values){/if}
    </div>
  {/if}
  {#if !stats}
    {#if !error}
      <p class="text-body text-fg-3">Loading…</p>
    {/if}
  {:else}
    <div class="grid grid-cols-1 gap-4 lg:grid-cols-4">
      <!-- Headline -->
      <div class="surface p-4 lg:col-span-1">
        <div class="text-meta text-fg-2 font-semibold">Total crops</div>
        <div class="mt-2 text-kpi font-semibold text-fg-1 tabular-nums">
          {fmt(stats.total_crops)}
        </div>
        <dl class="mt-4 space-y-1.5 text-meta">
          <div class="flex justify-between">
            <dt class="text-fg-2">Validated (ground truth)</dt>
            <dd class="text-success-text tabular-nums">{fmt(stats.validated)}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-fg-2">Test holdout</dt>
            <dd class="tabular-nums">{fmt(stats.test_holdout)}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-fg-2">Distinct sources</dt>
            <dd class="tabular-nums">{stats.by_source.length}</dd>
          </div>
        </dl>
      </div>

      <!-- Labeled breakdown -->
      <div class="surface p-4 lg:col-span-2">
        <header class="mb-3 flex items-center justify-between">
          <h3 class="text-body font-semibold text-fg-2">Labeled by source</h3>
          <span class="text-meta text-fg-3" data-testid="labeled-total"
            >{fmt(labeledTotal)} total</span
          >
        </header>
        <p class="mb-2 text-meta text-fg-3" data-testid="labeled-not-ground-truth">
          A label is not ground truth until a human validates it. Only {fmt(
            stats.validated,
          )} of {fmt(stats.total_crops)} crops are validated; VLM and detector labels are suggestions.
        </p>
        {#if labeledRows.length === 0}
          <p class="text-body text-fg-3">No labels yet.</p>
        {:else}
          <ul class="space-y-1.5">
            {#each labeledRows as row (row.key)}
              <li class="flex items-center gap-3 text-meta">
                <span class="w-24 shrink-0 text-fg-2">{row.label}</span>
                <div class="relative h-3 grow overflow-hidden rounded bg-surface-1">
                  <div
                    class="h-full {row.tone}"
                    style:width="{Math.max(0.5, row.pct)}%"
                    title="{row.pct}%"
                  ></div>
                </div>
                <span class="w-20 shrink-0 text-right text-fg-2 tabular-nums">
                  {fmt(row.count)}
                </span>
                <span class="w-12 shrink-0 text-right text-fg-3 tabular-nums">
                  {row.pct.toFixed(1)}%
                </span>
              </li>
            {/each}
          </ul>
        {/if}
      </div>

      <!-- Embedding (v0.4.0): items with and without a vector, as served. -->
      {#if stats.embedding}
        <div class="surface p-4 lg:col-span-1" data-testid="dataset-embedding">
          <h3 class="mb-3 text-body font-semibold text-fg-2">Embedding</h3>
          <dl class="space-y-1.5 text-meta">
            <div class="flex justify-between">
              <dt class="text-fg-2">Embedded</dt>
              <dd class="tabular-nums">{fmt(stats.embedding.embedded)}</dd>
            </div>
            <div class="flex justify-between">
              <dt class="text-fg-2">Not embedded</dt>
              <dd class="tabular-nums">{fmt(stats.embedding.not_embedded)}</dd>
            </div>
          </dl>
          {#if stateChips.length > 0}
            <div class="mt-2 flex flex-wrap gap-1" data-testid="embedding-by-state">
              {#each stateChips as chip (chip.state)}
                <span class="chip text-caption" title={chip.title}
                  >{chip.label} {fmt(chip.count)}</span
                >
              {/each}
            </div>
          {/if}
        </div>
      {/if}

      <!-- Last clustering -->
      <div class="surface p-4 lg:col-span-1">
        <h3 class="mb-3 text-body font-semibold text-fg-2">Last clustering</h3>
        <dl class="space-y-1.5 text-meta">
          <div class="flex justify-between">
            <dt class="text-fg-2">When</dt>
            <dd class="text-fg-2" title={stats.clusters.last_run_at ?? '—'}>
              {formatTimestamp(stats.clusters.last_run_at)}
            </dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-fg-2">Clusters (total now)</dt>
            <dd class="tabular-nums" data-testid="dataset-cluster-count">
              {fmt(stats.clusters.cluster_count)}
            </dd>
          </div>
          {#if stats.clusters.last_run_cluster_count != null}
            <div class="flex justify-between">
              <dt
                class="text-fg-2"
                title="Clusters the most recent run created. A run only clusters the residual pool (crops not already in a class cluster), so this is usually far smaller than the total."
              >
                Made by last run
              </dt>
              <dd class="tabular-nums">{fmt(stats.clusters.last_run_cluster_count)}</dd>
            </div>
          {/if}
          <div class="flex justify-between">
            <dt class="text-fg-2">Residual</dt>
            <dd class="tabular-nums">{fmt(stats.clusters.residual_count)}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-fg-2">Noise (cid&lt;0)</dt>
            <dd class="tabular-nums">{fmt(stats.clusters.noise_count)}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-fg-2">Method</dt>
            <dd class="text-fg-2">
              {stats.clusters.method ?? '—'}
            </dd>
          </div>
        </dl>
      </div>
    </div>

    <!-- Slot detections — separate from class labels. Titled from the
         active stats-capable slot's own spec. A detector runs on every
         crop and tries to find a region bbox; a segmenter is the
         fallback when it misses; humans place regions via the labeler
         UI. The denominator
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
            <h3 class="text-body font-semibold text-fg-2">{statsSpec.panelTitle}</h3>
            <span class="text-meta text-fg-3">
              {fmt(p?.boxed ?? 0)} with a box ·
              <span title="Accepted by the automatic verifier; not a human review">
                {fmt(p?.confirmed ?? 0)} verifier-confirmed
              </span>
            </span>
          </header>
          {#if slotRows.length === 0}
            <p class="text-body text-fg-3">No detections yet.</p>
          {:else}
            <ul class="space-y-1.5">
              {#each slotRows as row (row.key)}
                <li class="flex items-center gap-3 text-meta">
                  <span class="w-32 shrink-0 text-fg-2">{row.label}</span>
                  <div class="relative h-3 grow overflow-hidden rounded bg-surface-1">
                    <div
                      class="h-full {row.tone}"
                      style:width="{Math.max(0.5, row.pct)}%"
                      title="{row.pct}% of total crops"
                    ></div>
                  </div>
                  <span class="w-20 shrink-0 text-right text-fg-2 tabular-nums">
                    {fmt(row.count)}
                  </span>
                  <span class="w-12 shrink-0 text-right text-fg-3 tabular-nums">
                    {row.pct.toFixed(1)}%
                  </span>
                </li>
              {/each}
            </ul>
          {/if}
        </div>

        <!-- Coverage summary -->
        <div class="surface p-4">
          <h3 class="mb-3 text-body font-semibold text-fg-2">
            {statsSpec.coverageTitle}
          </h3>
          <div class="flex items-baseline gap-2">
            <span class="text-kpi text-fg-1 tabular-nums">
              {((p?.coverage_share ?? 0) * 100).toFixed(1)}%
            </span>
            <span class="text-meta text-fg-3">of crops have a box</span>
          </div>
          <!-- DQ-m11: this
               used to assert "(includes rejected/failed attempts)" about
               `total_detected` as fact. The field's own contract
               (DatasetStats.regions.total_detected in api.ts) says it
               should be a superset of `boxed` — but live, it equals
               `boxed` exactly, so verify-rejected and no-box crops that
               DID go through a detector aren't reflected in it. Rather than assert something the served
               number doesn't actually demonstrate, state what `boxed` is
               (the honest, verified count) and flag total_detected as a
               credit tally that may not include every rejected/no-box
               attempt on this deployment — not a confident inclusion
               claim. -->
          <p class="mt-2 text-meta text-fg-3">
            {fmt(p?.boxed ?? 0)} crops carry a box ({fmt(p?.confirmed ?? 0)} verifier-confirmed,
            not human-reviewed). The remaining
            {fmt(stats.total_crops - (p?.boxed ?? 0))} either had no visible detection (the
            visibility pre-filter said no), were rejected after detection, or the detectors
            haven't reached them yet.
            {fmt(p?.total_detected ?? 0)} crops carry detector credit (<code
              >regions.total_detected</code
            >) — this may not cover every rejected or box-less attempt.
          </p>
        </div>
      </div>
    {/if}

    <div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <!-- Unlabeled / pending -->
      <div class="surface p-4">
        <header class="mb-3 flex items-center justify-between">
          <h3 class="text-body font-semibold text-fg-2">Unlabeled</h3>
          <!-- The header used to sum pending_detection +
               pending_verification + no_label_source, but those buckets
               overlap (a crop can be pending region detection AND have no
               class), so the "total" could exceed total_crops. The server
               serves no combined total, so the header shows its own
               class-less count and the rows below say what each is. -->
          <span class="text-meta text-fg-3" data-testid="unlabeled-header-count"
            >{fmt(stats.unlabeled.no_label_source)} without a class</span
          >
        </header>
        <dl class="space-y-1.5 text-body">
          <div class="flex justify-between">
            <dt class="text-fg-2">Pending detection</dt>
            <dd class="text-warn-text tabular-nums">
              {fmt(stats.unlabeled.pending_detection)}
            </dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-fg-2">Pending verification</dt>
            <dd class="text-warn-text tabular-nums">
              {fmt(stats.unlabeled.pending_verification)}
            </dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-fg-2">No label source (class_id missing)</dt>
            <dd class="text-danger-text tabular-nums">
              {fmt(stats.unlabeled.no_label_source)}
            </dd>
          </div>
          <!-- #36 item 2: D1 fix — labeled.* now only counts docs with a
               real class_id, so a VLM attempt that couldn't resolve one
               shows up here instead of double-counting as "VLM-labeled". -->
          <div class="flex justify-between">
            <dt
              class="text-fg-2"
              title="VLM looked at the crop but didn't resolve a class"
            >
              VLM, no class
            </dt>
            <dd class="text-danger-text tabular-nums">
              {fmt(stats.unlabeled.vlm_no_class)}
            </dd>
          </div>
          <!-- Detector-proposed crops nothing has classified yet,
               a subset of the class-less count above (served here
               instead of the old, always-0 labeled.by_proposal). -->
          <div class="flex justify-between">
            <dt
              class="text-fg-2"
              title="A detector proposed this crop as an object, but nothing has classified it yet (a subset of the class-less count)"
            >
              Detector proposal, no class
            </dt>
            <dd class="text-danger-text tabular-nums">
              {fmt(stats.unlabeled.by_proposal)}
            </dd>
          </div>
        </dl>
        <p class="mt-2 text-meta text-fg-3">
          Pending detection / verification count region-worklog state, not class labels,
          so they can overlap the class-less count.
        </p>
      </div>

      <!-- In-progress queue + ETA -->
      <div class="surface p-4">
        <h3 class="mb-3 text-body font-semibold text-fg-2">In-flight pipeline</h3>
        <div class="flex items-baseline gap-3">
          <span class="text-kpi text-fg-1 tabular-nums">
            {fmt(stats.in_progress.region_drain_total_unfinished)}
          </span>
          <span class="text-body text-fg-2">crops awaiting detection / verification</span>
        </div>

        {#if stats.in_progress.region_stall_reason}
          <!-- V-1: the server says why the drain can't progress (a region
               dependency is down); shown verbatim. -->
          <p
            class="mt-3 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-meta text-warn-text"
            data-testid="region-stall-reason"
          >
            Stalled: {stats.in_progress.region_stall_reason}
          </p>
        {/if}
        {#if stats.in_progress.region_drain_total_unfinished > 0}
          <!-- ETA — rolling drain rate over last ~60s. Caveat: only as
               good as the steady-state assumption (sam-worker bursts
               look like big rate spikes; queue refills look like
               regressions). Surface the raw rate alongside the ETA so
               the operator can sanity-check. -->
          <dl class="mt-3 space-y-1 text-meta">
            <div class="flex items-baseline justify-between">
              <dt class="text-fg-2">ETA to drain</dt>
              <dd class="text-fg-1 tabular-nums">
                {#if etaSeconds == null}
                  <span class="text-fg-3" title="Need ≥10s + 2 samples to compute">
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
              <dt class="text-fg-2">Drain rate</dt>
              <dd class="tabular-nums">
                {#if drainRate == null}
                  <span class="text-fg-3">…</span>
                {:else if drainRate < 0}
                  <span class="text-danger-text">+{formatRate(-drainRate)} (growing)</span
                  >
                {:else if drainRate === 0}
                  <span class="text-fg-3">0 (stalled)</span>
                {:else}
                  <span class="text-accent-text">{formatRate(drainRate)}</span>
                {/if}
              </dd>
            </div>
          </dl>

          <!-- Indeterminate "the queue is draining" bar -->
          <div class="mt-3 h-1.5 overflow-hidden rounded bg-surface-1">
            <div class="h-full w-1/3 animate-pulse rounded bg-accent"></div>
          </div>
        {:else}
          <p class="mt-3 text-meta text-success-text">queue drained</p>
        {/if}
        <p class="mt-3 text-meta text-fg-3">
          Matches the API's <code>ingest/region_drain</code> total. While &gt; 0 the ingest
          walker waits before triggering the next clustering pass.
        </p>
      </div>
    </div>
  {/if}
</section>
