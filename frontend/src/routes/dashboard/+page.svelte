<script lang="ts">
  /*
   * Pipeline dashboard route — the app's single home page.
   *
   * Merged 2026-09 from the legacy `/` page (recent-crops + quick-actions
   * + a per-class balance chart) into this one, since `/dashboard` was
   * already the live/current pipeline view and having two overlapping
   * "home" pages was confusing. Nothing from the legacy page was
   * dropped — see below for where each piece landed:
   *
   *   1. `AutoLabelPanel` / `DatasetStats` — unchanged, already here.
   *   2. Quick actions (Run Gemma / Export / Snapshot) — brought over
   *      verbatim, same handlers.
   *   3. Class balance bar chart — brought over verbatim; this is a
   *      genuinely different view from DatasetStats' "Labeled by
   *      source" breakdown (per-CLASS, not per-source), so it's kept
   *      as its own section rather than folded into DatasetStats.
   *   4. Recent labels grid — brought over verbatim.
   *
   * The legacy Ingestion panel (Total crops/Validated/Test holdout/
   * Images processed/Pending/Last run) was NOT brought over — every one
   * of those numbers is already shown live (SSE, not a 30s poll) in
   * DatasetStats' "Pipeline stats" and "In-flight pipeline" sections.
   * Duplicating it here would just be two numbers disagreeing during
   * the gap between polls.
   */
  import { apiBase } from '$lib/api';
  import { healthStore } from '$stores/health.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import AutoLabelPanel from '$components/AutoLabelPanel.svelte';
  import DatasetStats from '$components/DatasetStats.svelte';
  import {
    exportYolo,
    getCrops,
    getStats,
    getThumbUrl,
    pollAutoLabelJob,
    runVlmOnCluster,
    type AutoLabelJobState,
  } from '$lib/api';
  import type { Crop, ExportResult, StatsSummary } from '$lib/types';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('dashboard');
  });

  // ---- Class balance + recent labels (from the legacy `/` page) --------

  let legacyStats = $state<StatsSummary | null>(null);
  let recent = $state<Crop[]>([]);
  let legacyLoading = $state<boolean>(false);

  let vlmOpen = $state<boolean>(false);
  let vlmClusterId = $state<string>('');
  let vlmBusy = $state<boolean>(false);
  // Live status while the cluster-scoped VLM job runs — polled via
  // pollAutoLabelJob (POST /vlm/label_cluster/{id}, then GET
  // /pipeline/auto_label/status), same job shape AutoLabelPanel shows
  // for a full recluster.
  let vlmJob = $state<AutoLabelJobState | null>(null);

  // M13 (2026-09-24 interactive pass): "Export Dataset (YOLO)" used to
  // fire POST {API_PREFIX}/export/yolo with no confirmation and toast
  // "job started" — but that endpoint is synchronous (verified live and
  // against openprocessor's export_yolo handler: it `await`s the full
  // 5-step export pipeline before returning), so the toast lied about
  // what had actually happened by the time it appeared. A confirm step
  // now gates the click (the export can take a while and touches the
  // frozen dataset), and the modal renders the real returned fields
  // once the request resolves rather than assuming a job id exists.
  let exportConfirmOpen = $state<boolean>(false);
  let exportRunning = $state<boolean>(false);
  let exportResult = $state<ExportResult | null>(null);
  let exportError = $state<string | null>(null);

  async function refreshLegacy(): Promise<void> {
    legacyLoading = true;
    try {
      const [s, c] = await Promise.allSettled([
        getStats(),
        getCrops({
          label_validated: true,
          sort: 'updated_at:desc',
          limit: 20,
          page: 1,
        }),
      ]);
      legacyStats = s.status === 'fulfilled' ? s.value : null;
      recent = c.status === 'fulfilled' ? c.value.items : [];
    } finally {
      legacyLoading = false;
    }
  }

  $effect(() => {
    void refreshLegacy();
    const id = setInterval(() => void refreshLegacy(), 30_000);
    return () => clearInterval(id);
  });

  async function runVlm(): Promise<void> {
    const id = Number(vlmClusterId);
    if (!Number.isFinite(id) || id < 0) {
      toastStore.error('Enter a valid cluster id');
      return;
    }
    vlmBusy = true;
    vlmJob = null;
    try {
      vlmJob = await runVlmOnCluster(id);
      const final = await pollAutoLabelJob((j) => (vlmJob = j));
      const stages = (final.result?.stages ?? {}) as Record<
        string,
        Record<string, unknown>
      >;
      const vlm = stages.vlm ?? {};
      if (final.status === 'failed') {
        toastStore.error(`VLM run failed: ${final.error ?? 'unknown error'}`);
      } else {
        toastStore.success(
          `VLM labeled ${Number(vlm.predicted ?? 0)} crops (${Number(vlm.updated ?? 0)} updated).`,
        );
        vlmOpen = false;
      }
    } catch (e) {
      toastStore.error(`VLM run failed: ${(e as Error).message}`);
    } finally {
      vlmBusy = false;
    }
  }

  function openExportConfirm(): void {
    exportResult = null;
    exportError = null;
    exportConfirmOpen = true;
  }

  async function runExport(): Promise<void> {
    exportRunning = true;
    exportResult = null;
    exportError = null;
    try {
      // Synchronous — the response IS the finished export, not a
      // queued-job acknowledgement. Render it directly.
      exportResult = await exportYolo();
      toastStore.success(
        `Export complete: ${exportResult.export_dir ?? exportResult.status}`,
      );
    } catch (e) {
      exportError = (e as Error).message;
      toastStore.error(`Export failed: ${exportError}`);
    } finally {
      exportRunning = false;
    }
  }

  const balance = $derived.by(() => {
    if (!legacyStats?.per_class) return [];
    const max = Math.max(1, ...legacyStats.per_class.map((c) => c.validated_count));
    return [...legacyStats.per_class]
      .sort((a, b) => b.validated_count - a.validated_count)
      .slice(0, 30)
      .map((c) => ({
        ...c,
        pct: Math.max(2, Math.round((c.validated_count / max) * 100)),
        tier: c.adequacy ?? 'block',
      }));
  });

  function tierBg(t: string): string {
    if (t === 'ok') return 'bg-green-500';
    if (t === 'warn') return 'bg-orange-500';
    return 'bg-red-500';
  }
</script>

<div class="mx-auto max-w-7xl space-y-6 p-6">
  <header class="flex items-end justify-between">
    <div>
      <h1 class="text-2xl font-semibold tracking-tight">Dashboard</h1>
      <p class="text-sm text-zinc-500">
        Pipeline stats and manual clustering control. Stats refresh every 10s.
      </p>
    </div>
  </header>

  {#if !healthStore.ok}
    <div
      class="rounded-md border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200"
    >
      <strong>API unavailable.</strong> Check that the OpenProcessor API is reachable from
      <code class="font-mono"
        >{apiBase || (typeof window !== 'undefined' ? window.location.host : '')}</code
      >.
    </div>
  {/if}

  <!-- Quick actions -->
  <section class="surface flex flex-wrap items-center gap-2 p-4">
    <h2 class="mr-3 text-sm font-semibold text-zinc-400">Quick actions</h2>
    <button class="btn btn-primary" type="button" onclick={() => (vlmOpen = true)}>
      Run VLM Labeling
    </button>
    <button class="btn" type="button" onclick={openExportConfirm}>
      Export Dataset (YOLO)
    </button>
    <span class="grow"></span>
    <button
      class="btn"
      type="button"
      onclick={() => void refreshLegacy()}
      disabled={legacyLoading}
    >
      {legacyLoading ? 'Refreshing...' : 'Refresh'}
    </button>
  </section>

  <AutoLabelPanel />

  <DatasetStats />

  <!-- Class balance -->
  <section class="surface p-4">
    <header class="mb-3 flex items-center justify-between">
      <h2 class="text-sm font-semibold text-zinc-300">Class balance (validated)</h2>
      <!-- m6 (2026-09-24 interactive pass): the legend used to hardcode
           500/100, contradicting the served thresholds
           (block_below/warn_below on legacyStats.thresholds, the same
           values the bar colors are already computed from). Render the
           real numbers, or nothing once loaded but absent, rather than
           a number that never matches the bars. -->
      <span class="text-xs text-zinc-500">
        {#if legacyStats?.thresholds}
          green &ge;{legacyStats.thresholds.warn_below} · orange {legacyStats.thresholds
            .block_below}–{legacyStats.thresholds.warn_below - 1} · red &lt;{legacyStats
            .thresholds.block_below}
        {/if}
      </span>
    </header>

    {#if !legacyStats}
      <p class="text-sm text-zinc-500">Loading...</p>
    {:else if balance.length === 0}
      <p class="text-sm text-zinc-500">No classes yet.</p>
    {:else}
      <ul class="space-y-1.5">
        {#each balance as row (row.class_id)}
          <li class="flex items-center gap-3 text-xs">
            <span class="w-32 shrink-0 truncate text-zinc-300" title={row.class_name}>
              {row.class_name}
            </span>
            <div class="relative h-4 grow overflow-hidden rounded bg-zinc-900">
              <div class="h-full {tierBg(row.tier)}" style:width="{row.pct}%"></div>
            </div>
            <span class="w-16 shrink-0 text-right font-mono text-zinc-400">
              {row.validated_count}
            </span>
          </li>
        {/each}
      </ul>
    {/if}
  </section>

  <!-- Recent activity -->
  <section class="surface p-4">
    <h2 class="mb-3 text-sm font-semibold text-zinc-300">Recent labels (last 20)</h2>
    {#if recent.length === 0}
      <p class="text-sm text-zinc-500">No recent labels.</p>
    {:else}
      <ul class="grid grid-cols-3 gap-3 sm:grid-cols-5 lg:grid-cols-10">
        {#each recent as crop (crop.id)}
          <li class="rounded-md border border-zinc-800 bg-zinc-900">
            <img
              src={getThumbUrl(crop.id)}
              alt="crop"
              loading="lazy"
              class="aspect-square w-full rounded-t-md object-contain"
            />
            <div
              class="truncate px-1.5 py-1 text-[10px] text-zinc-300"
              title={crop.class_name ?? ''}
            >
              {crop.class_name ?? '—'}
            </div>
          </li>
        {/each}
      </ul>
    {/if}
  </section>
</div>

{#if vlmOpen}
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-3 text-base font-semibold">Run VLM on cluster</h3>
      <label class="mb-3 block text-sm">
        <span class="mb-1 block text-zinc-400">Cluster ID</span>
        <input
          type="number"
          min="0"
          bind:value={vlmClusterId}
          class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
          placeholder="e.g. 42"
        />
      </label>
      <p class="mb-4 text-xs text-zinc-500">
        Only un-validated crops in the cluster will be sent. Test-holdout crops are
        excluded by the API.
      </p>
      {#if vlmJob}
        <p class="mb-4 text-xs text-zinc-400">
          {vlmJob.status === 'running'
            ? `Running — stage: ${vlmJob.stage || 'preparing…'}${vlmJob.total > 0 ? ` (${vlmJob.processed}/${vlmJob.total})` : ''}`
            : `Status: ${vlmJob.status}`}
        </p>
      {/if}
      <div class="flex justify-end gap-2">
        <button
          type="button"
          class="btn"
          onclick={() => (vlmOpen = false)}
          disabled={vlmBusy}
        >
          Cancel
        </button>
        <button type="button" class="btn btn-primary" onclick={runVlm} disabled={vlmBusy}>
          {vlmBusy ? 'Running...' : 'Run'}
        </button>
      </div>
    </div>
  </div>
{/if}

{#if exportConfirmOpen}
  <!-- M13 (2026-09-24 interactive pass): confirm before kicking off a
       synchronous full-dataset YOLO export, and show the served result
       (or error) once it resolves — no "job started" fiction. -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-3 text-base font-semibold">Export dataset (YOLO)</h3>
      {#if !exportResult && !exportError}
        <p class="mb-4 text-xs text-zinc-500">
          Runs the full YOLO export pipeline synchronously (resize, stratified split
          honoring the frozen test holdout, manifest write) — this can take a while and
          the page will wait for it to finish.
        </p>
      {/if}
      {#if exportRunning}
        <p class="mb-4 text-xs text-zinc-400">Exporting… this may take a few minutes.</p>
      {/if}
      {#if exportError}
        <p class="mb-4 text-xs text-red-300">Export failed: {exportError}</p>
      {/if}
      {#if exportResult}
        <dl class="mb-4 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
          <dt class="text-zinc-500">Status</dt>
          <dd class="font-mono text-zinc-200">{exportResult.status}</dd>
          {#if exportResult.export_dir}
            <dt class="text-zinc-500">Export dir</dt>
            <dd class="break-all font-mono text-zinc-200">{exportResult.export_dir}</dd>
          {/if}
          {#if exportResult.dataset_sha}
            <dt class="text-zinc-500">Dataset SHA</dt>
            <dd class="break-all font-mono text-zinc-200">{exportResult.dataset_sha}</dd>
          {/if}
          {#if exportResult.split_counts}
            <dt class="text-zinc-500">Split counts</dt>
            <dd class="font-mono text-zinc-200">
              {Object.entries(exportResult.split_counts)
                .map(([k, v]) => `${k}: ${v}`)
                .join(' · ')}
            </dd>
          {/if}
          {#if exportResult.finished_at}
            <dt class="text-zinc-500">Finished</dt>
            <dd class="font-mono text-zinc-200">{exportResult.finished_at}</dd>
          {/if}
        </dl>
      {/if}
      <div class="flex justify-end gap-2">
        <button
          type="button"
          class="btn"
          onclick={() => (exportConfirmOpen = false)}
          disabled={exportRunning}
        >
          {exportResult || exportError ? 'Close' : 'Cancel'}
        </button>
        {#if !exportResult && !exportError}
          <button
            type="button"
            class="btn btn-primary"
            onclick={runExport}
            disabled={exportRunning}
          >
            {exportRunning ? 'Exporting…' : 'Export'}
          </button>
        {/if}
      </div>
    </div>
  </div>
{/if}
