<script lang="ts">
  /**
   * /export: dataset stats per class, the test-holdout freeze, and the
   * YOLO export. The export run and the freeze live in `$lib/export/`
   * controllers; the sections in `$lib/components/export/` render them.
   */
  import { apiErrorText } from '$lib/api';
  import {
    apiBase,
    scoped,
    exportStatus,
    getStats,
    getTestHoldoutStats,
    listDatasets,
  } from '$lib/api';
  import type { StatsSummary, TestHoldoutStats } from '$lib/types';
  import {
    buildExportRows,
    registryArtifactsAvailable,
  } from '$lib/export/exportDatasetRows';
  import { createExportRunController } from '$lib/export/exportRunController.svelte';
  import { createFreezeHoldoutController } from '$lib/export/freezeHoldoutController.svelte';
  import ValidatedRatioNotice from '$lib/components/labelConfirmation/ValidatedRatioNotice.svelte';
  import HoldoutStatusCard from '$components/export/HoldoutStatusCard.svelte';
  import ExportClassTable from '$components/export/ExportClassTable.svelte';
  import ExportControls from '$components/export/ExportControls.svelte';
  import ExportRunModal from '$components/export/ExportRunModal.svelte';
  import FreezeHoldoutModal from '$components/export/FreezeHoldoutModal.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';

  $effect(() => {
    keyboardStore.setScope('export');
  });

  // ---- data --------------------------------------------------------------

  let stats = $state<StatsSummary | null>(null);
  let holdout = $state<TestHoldoutStats | null>(null);
  let loading = $state<boolean>(false);
  let error = $state<string | null>(null);

  const run = createExportRunController(loadAll);
  const freeze = createFreezeHoldoutController(loadAll);
  // m15 (2026-09-24 interactive pass): the registry download buttons used
  // to gate on `exportState?.status === 'success'` alone, which is a
  // single shared job-status slot (see m28's dashboard finding) and
  // stayed "success" even with no *current* frozen multi-class (`yolo`)
  // export on disk — so the buttons looked enabled and 404ed. Gated on
  // the served `GET {API_PREFIX}/export/datasets` list instead.
  let hasMulticlassExport = $state<boolean>(false);

  async function loadAll(): Promise<void> {
    loading = true;
    error = null;
    try {
      const [s, h, e, d] = await Promise.allSettled([
        getStats(),
        getTestHoldoutStats(),
        exportStatus(),
        listDatasets({ kind: 'yolo' }),
      ]);
      stats = s.status === 'fulfilled' ? s.value : null;
      holdout = h.status === 'fulfilled' ? h.value : null;
      run.exportState = e.status === 'fulfilled' ? e.value : null;
      hasMulticlassExport = registryArtifactsAvailable(
        d.status === 'fulfilled' ? d.value.datasets : null,
        run.exportState,
      );
      if (s.status === 'rejected' && h.status === 'rejected') {
        error = 'API unavailable';
      }
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    void loadAll();
    return () => {
      run.dispose();
    };
  });

  // ---- dataset rows ------------------------------------------------------

  const rows = $derived(buildExportRows(stats?.per_class, holdout));

  // ---- HDD source distribution ------------------------------------------

  interface HddBucket {
    key: string;
    doc_count: number;
  }
  let hddSources = $state<HddBucket[]>([]);
  // m16 (2026-09-24 interactive pass): this fetch used to fail silently —
  // a 503/non-JSON response just left hddSources empty with no visible
  // sign anything had gone wrong, so the "totals failed" case looked
  // identical to "no source data at all".
  let hddSourcesError = $state<string | null>(null);
  // Pull from the same {API_PREFIX}/stats/dataset response — the existing `getStats`
  // surface only exposes per_class summary; we hit the
  // dataset-stats endpoint directly via fetch for the by_source bucket.
  // The thin {API_PREFIX}/stats/dataset response carries an `by_source` HDD bucket
  // array which `getStats` (typed to StatsSummary) doesn't surface. Hit it
  // directly so we can render the source-distribution chip row.
  $effect(() => {
    void (async () => {
      try {
        const res = await fetch(`${apiBase}${scoped()}/stats/dataset`, {
          method: 'GET',
        });
        if (!res.ok) {
          hddSourcesError = `Dataset totals unavailable (API ${res.status}) — by-source breakdown may be stale or missing.`;
          return;
        }
        const ct = res.headers.get('content-type') ?? '';
        if (!ct.includes('application/json')) {
          hddSourcesError = 'Dataset totals unavailable (non-JSON response).';
          return;
        }
        const json = (await res.json()) as { by_source?: HddBucket[]; error?: string };
        if (json.error) {
          hddSourcesError = `Dataset totals unavailable: ${json.error}`;
          return;
        }
        hddSourcesError = null;
        if (Array.isArray(json.by_source)) hddSources = json.by_source;
      } catch (e) {
        hddSourcesError = `Dataset totals unavailable: ${apiErrorText(e)}`;
      }
    })();
  });
</script>

<div class="mx-auto flex h-full max-w-7xl flex-col gap-4 p-6">
  <header class="flex flex-wrap items-center gap-3">
    <h1 class="text-2xl font-semibold tracking-tight">Export dataset</h1>
    <span class="grow"></span>
    <button class="btn" type="button" onclick={() => void loadAll()} disabled={loading}>
      {loading ? 'Refreshing…' : 'Refresh'}
    </button>
  </header>

  {#if stats && stats.dataset_error == null}
    <ValidatedRatioNotice validated={stats.validated_crops} total={stats.total_crops} />
  {/if}

  <HoldoutStatusCard {holdout} {rows} onFreeze={freeze.openFreeze} />

  <!-- HDD source distribution -->
  {#if hddSourcesError}
    <section
      class="surface border-orange-500/40 bg-orange-500/10 p-3 text-xs text-orange-200"
    >
      {hddSourcesError}
    </section>
  {:else if hddSources.length > 0}
    <section class="surface p-4">
      <h2 class="mb-2 text-sm font-semibold text-zinc-300">Source distribution</h2>
      <ul class="flex flex-wrap gap-2 text-xs">
        {#each hddSources as src (src.key)}
          <li class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1">
            <span class="font-mono text-zinc-300">{src.key}</span>
            <span class="ml-1 text-zinc-500">{src.doc_count.toLocaleString()}</span>
          </li>
        {/each}
      </ul>
    </section>
  {/if}

  <ExportClassTable
    {rows}
    {loading}
    {error}
    blockBelow={stats?.thresholds?.block_below ?? null}
  />

  <ExportControls {run} {hasMulticlassExport} />
</div>

<ExportRunModal {run} />

<FreezeHoldoutModal {freeze} />
