<script lang="ts">
  /*
   * Pipeline-stats dashboard panel.
   *
   * Polls `GET /curation/stats/dataset` every 10s and renders:
   *   - Total crops (large headline)
   *   - Labeled-by-source table with proportion bars
   *   - Unlabeled breakdown (pending_detection / pending_verification /
   *     no_label_source)
   *   - In-progress queue (sam_drain_total_unfinished)
   *   - Last clustering run summary (timestamp, method, cluster_count,
   *     residual_count, noise_count)
   *
   * The poll is skipped while a previous fetch is still in flight so a
   * slow OS query never stacks requests. The component is paused while
   * `document.visibilityState !== 'visible'` to avoid burning tokens
   * when the tab is backgrounded.
   */
  import { onDestroy } from "svelte";
  import { getDatasetStats, type DatasetStats } from "$lib/api";

  interface Props {
    /** Poll interval in ms. Default 10s — match the spec. */
    pollMs?: number;
  }
  let { pollMs = 10_000 }: Props = $props();

  let stats = $state<DatasetStats | null>(null);
  let error = $state<string | null>(null);
  let lastUpdated = $state<number | null>(null);
  let inFlight = $state<boolean>(false);
  let timer: ReturnType<typeof setInterval> | null = null;

  async function refresh(): Promise<void> {
    if (inFlight) return;
    inFlight = true;
    try {
      stats = await getDatasetStats();
      error = null;
      lastUpdated = Date.now();
    } catch (e) {
      error = (e as Error).message || "failed to load stats";
    } finally {
      inFlight = false;
    }
  }

  function start(): void {
    if (timer != null) return;
    void refresh();
    timer = setInterval(() => {
      if (document.visibilityState === "visible") void refresh();
    }, pollMs);
  }

  function stop(): void {
    if (timer != null) {
      clearInterval(timer);
      timer = null;
    }
  }

  $effect(() => {
    start();
    const onVis = (): void => {
      if (document.visibilityState === "visible") void refresh();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      document.removeEventListener("visibilitychange", onVis);
      stop();
    };
  });

  onDestroy(stop);

  const labeledTotal = $derived.by(() => {
    const l = stats?.labeled;
    if (!l) return 0;
    return l.by_human + l.by_gemma + l.by_v6 + l.by_lpr + l.other;
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
        key: "by_human",
        label: "Human",
        count: l.by_human,
        tone: "bg-green-500",
      },
      {
        key: "by_gemma",
        label: "Gemma",
        count: l.by_gemma,
        tone: "bg-blue-500",
      },
      {
        key: "by_v6",
        label: "v6 model",
        count: l.by_v6,
        tone: "bg-purple-500",
      },
      { key: "by_lpr", label: "LPR", count: l.by_lpr, tone: "bg-amber-500" },
      { key: "other", label: "Other", count: l.other, tone: "bg-zinc-500" },
    ];
    return rows.map((r) => ({
      ...r,
      pct: Math.round((r.count / labeledTotal) * 1000) / 10,
    }));
  });

  const unlabeledTotal = $derived.by(() => {
    const u = stats?.unlabeled;
    if (!u) return 0;
    return u.pending_detection + u.pending_verification + u.no_label_source;
  });

  function fmt(n: number | undefined | null): string {
    return (n ?? 0).toLocaleString();
  }

  function fmtTimestamp(iso: string | null): string {
    if (!iso) return "—";
    try {
      return new Date(iso).toLocaleString();
    } catch {
      return iso;
    }
  }

  function fmtRelative(ms: number | null): string {
    if (ms == null) return "—";
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
      <span title={lastUpdated ? new Date(lastUpdated).toLocaleString() : ""}>
        updated {fmtRelative(lastUpdated)}
      </span>
      <button
        type="button"
        class="btn"
        onclick={() => void refresh()}
        disabled={inFlight}
        aria-label="Refresh stats"
      >
        {inFlight ? "…" : "Refresh"}
      </button>
    </div>
  </header>

  {#if error && !stats}
    <div
      class="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200"
    >
      Stats unavailable: {error}
    </div>
  {:else if !stats}
    <p class="text-sm text-zinc-500">Loading…</p>
  {:else}
    <div class="grid grid-cols-1 gap-4 lg:grid-cols-4">
      <!-- Headline -->
      <div class="surface p-4 lg:col-span-1">
        <div class="text-xs uppercase tracking-wide text-zinc-500">
          Total crops
        </div>
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
                <div
                  class="relative h-3 grow overflow-hidden rounded bg-zinc-900"
                >
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
        <h3 class="mb-3 text-sm font-semibold text-zinc-300">
          Last clustering
        </h3>
        <dl class="space-y-1.5 text-xs">
          <div class="flex justify-between">
            <dt class="text-zinc-400">When</dt>
            <dd class="text-zinc-300" title={stats.clusters.last_run_at ?? "—"}>
              {fmtTimestamp(stats.clusters.last_run_at)}
            </dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-zinc-400">Clusters</dt>
            <dd class="font-mono">{fmt(stats.clusters.cluster_count)}</dd>
          </div>
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
              {stats.clusters.method ?? "—"}
            </dd>
          </div>
        </dl>
      </div>
    </div>

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

      <!-- In-progress queue -->
      <div class="surface p-4">
        <h3 class="mb-3 text-sm font-semibold text-zinc-300">
          In-flight pipeline
        </h3>
        <div class="flex items-baseline gap-3">
          <span class="font-mono text-2xl text-zinc-100">
            {fmt(stats.in_progress.sam_drain_total_unfinished)}
          </span>
          <span class="text-sm text-zinc-400">crops awaiting SAM3 / Gemma</span>
        </div>
        {#if stats.in_progress.sam_drain_total_unfinished > 0}
          <!-- Indeterminate "the queue is draining" bar -->
          <div class="mt-3 h-1.5 overflow-hidden rounded bg-zinc-900">
            <div class="h-full w-1/3 animate-pulse rounded bg-blue-500"></div>
          </div>
        {:else}
          <p class="mt-3 text-xs text-green-400">queue drained</p>
        {/if}
        <p class="mt-3 text-xs text-zinc-500">
          Matches <code class="font-mono">/curation/ingest/sam_drain</code> total. While
          &gt; 0 the ingest walker waits before triggering the next clustering pass.
        </p>
      </div>
    </div>
  {/if}
</section>
