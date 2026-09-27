<script lang="ts">
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';
  import {
    apiBase,
    scoped,
    exportStatus,
    exportYolo,
    freezeTestHoldout,
    getClassRegistryUrl,
    getDataYamlUrl,
    getManifestUrl,
    getStats,
    getTestHoldoutStats,
    listDatasets,
  } from '$lib/api';
  import type { ExportStatus, StatsSummary, TestHoldoutStats } from '$lib/types';
  import {
    buildExportRows,
    GAP_COLUMN_TITLE,
    gapCellTitle,
    TRAINABLE_COLUMN_TITLE,
    registryArtifactsAvailable,
    isNothingExportable,
    splitExportClasses,
    type ExportRow,
  } from '$lib/export/exportDatasetRows';
  import { formatCount } from '$lib/formatCount';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('export');
  });

  // ---- data --------------------------------------------------------------

  let stats = $state<StatsSummary | null>(null);
  let holdout = $state<TestHoldoutStats | null>(null);
  let loading = $state<boolean>(false);
  let error = $state<string | null>(null);

  // Sort
  type SortKey =
    | 'class_name'
    | 'class_id'
    | 'total'
    | 'validated'
    | 'trainable'
    | 'aug_target'
    | 'trainableGap';
  let sortKey = $state<SortKey>('trainableGap');
  let sortDir = $state<'asc' | 'desc'>('desc');

  // Export
  let versionTag = $state<string>('');
  // OpenProcessor 4c9499a: opt-in — drop any exported image that still
  // has an unlabeled object on it, rather than teaching the detector to
  // treat that object as background.
  let requireFullyLabeled = $state<boolean>(false);
  let exportRunning = $state<boolean>(false);
  let exportState = $state<ExportStatus | null>(null);
  let pollHandle: ReturnType<typeof setInterval> | null = null;
  let exportModalOpen = $state<boolean>(false);
  // m15 (2026-09-24 interactive pass): the registry download buttons used
  // to gate on `exportState?.status === 'success'` alone, which is a
  // single shared job-status slot (see m28's dashboard finding) and
  // stayed "success" even with no *current* frozen multi-class (`yolo`)
  // export on disk — so the buttons looked enabled and 404ed. Gated on
  // the served `GET {API_PREFIX}/export/datasets` list instead.
  let hasMulticlassExport = $state<boolean>(false);

  // Test holdout freeze
  let freezeOpen = $state<boolean>(false);
  let freezePercent = $state<number>(10);
  let freezeBusy = $state<boolean>(false);

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
      exportState = e.status === 'fulfilled' ? e.value : null;
      hasMulticlassExport = registryArtifactsAvailable(
        d.status === 'fulfilled' ? d.value.datasets : null,
        exportState,
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
      if (pollHandle) clearInterval(pollHandle);
      pollHandle = null;
    };
  });

  // ---- dataset rows ------------------------------------------------------

  const rows = $derived.by((): ExportRow[] => {
    const list = buildExportRows(stats?.per_class, holdout);
    list.sort((a, b) => {
      const dir = sortDir === 'asc' ? 1 : -1;
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir;
      return String(av).localeCompare(String(bv)) * dir;
    });
    return list;
  });
  // Classes with any validated or held-out crop come first; the rest (often
  // most of the registry) fold into one expandable row, so the classes that
  // actually have data are never buried under empty ones.
  const activeRows = $derived(rows.filter((r) => r.validated > 0 || r.test_count > 0));
  const emptyRows = $derived(rows.filter((r) => !(r.validated > 0 || r.test_count > 0)));
  let showEmptyClasses = $state(false);

  // DQ-M9 frontend half (docs/design/data-quality-pass-2026-09-24.md): the
  // Export button used to be enabled unconditionally — the audit's repro
  // was every class at the served "block" adequacy tier (0
  // class_validated dataset-wide) with Export still clickable, since
  // `POST /export/yolo` itself has no readiness gate. `loading` guards the
  // window before `stats` has ever arrived, where `rows` is legitimately
  // `[]` — don't flash "nothing to export" before the served data is in.
  const exportClassSplit = $derived(
    splitExportClasses(exportState?.class_split_counts ?? []),
  );

  const nothingExportable = $derived(!loading && isNothingExportable(rows));

  function setSort(k: SortKey): void {
    if (sortKey === k) {
      sortDir = sortDir === 'asc' ? 'desc' : 'asc';
    } else {
      sortKey = k;
      sortDir = k === 'class_name' ? 'asc' : 'desc';
    }
  }

  function gapClass(gap: number): string {
    if (gap <= 0) return 'bg-green-500/20 text-green-200 border-green-500/40';
    if (gap <= 200) return 'bg-orange-500/20 text-orange-200 border-orange-500/40';
    return 'bg-red-500/20 text-red-200 border-red-500/40';
  }

  function testBadge(deficient: boolean): string {
    return deficient
      ? 'bg-red-500/20 text-red-200 border-red-500/40'
      : 'bg-zinc-800 text-zinc-300 border-zinc-700';
  }

  // m16 (2026-09-24 interactive pass): the served per-class `adequacy`
  // tier wasn't rendered anywhere on this page. Just a display of the
  // server's own value — no client-side threshold logic.
  function adequacyClass(adequacy: string | null): string {
    if (adequacy === 'block') return 'bg-red-500/20 text-red-200 border-red-500/40';
    if (adequacy === 'warn')
      return 'bg-orange-500/20 text-orange-200 border-orange-500/40';
    if (adequacy === 'ok') return 'bg-green-500/20 text-green-200 border-green-500/40';
    return 'bg-zinc-800 text-zinc-400 border-zinc-700';
  }

  // ---- export ------------------------------------------------------------

  function startPolling(): void {
    if (pollHandle) return;
    pollHandle = setInterval(async () => {
      try {
        const s = await exportStatus();
        exportState = s;
        if (s.status !== 'running' && s.status !== 'pending') {
          exportRunning = false;
          if (pollHandle) {
            clearInterval(pollHandle);
            pollHandle = null;
          }
          if (s.status === 'success') {
            toastStore.success(`Export complete: ${s.export_dir ?? 'see manifest'}`);
            await loadAll();
          } else if (s.status === 'failed') {
            toastStore.error(`Export failed: ${s.error ?? 'unknown'}`);
          }
        }
      } catch (e) {
        toastStore.warn(`Status poll failed: ${(e as Error).message}`);
      }
    }, 5000);
  }

  // `POST {API_PREFIX}/export/yolo` is synchronous (see `ExportResult`'s
  // doc comment in `$lib/types` — the OpenProcessor handler `await`s the
  // export before responding, there is no queued-job ack) — the response
  // IS the finished export, not a "started" acknowledgement. Render its
  // own served `status` directly instead of assuming `running` and
  // polling; only fall back to polling when the backend itself reports
  // the job still `running`/`pending` (a future async backend, or a slow
  // one that didn't finish inline).
  async function runExport(): Promise<void> {
    exportRunning = true;
    exportModalOpen = true;
    try {
      const res = await exportYolo({
        version_tag: versionTag.trim() || undefined,
        require_fully_labeled_images: requireFullyLabeled,
      });
      exportState = {
        status: res.status,
        last_run: res.finished_at ?? res.started_at ?? new Date().toISOString(),
        export_dir: res.export_dir ?? null,
        error: res.status === 'failed' ? (res.message ?? 'unknown error') : null,
        message: res.message ?? null,
        image_count: res.image_count ?? null,
        object_count: res.object_count ?? null,
        split_object_counts:
          (res.split_object_counts as ExportStatus['split_object_counts']) ?? null,
        require_fully_labeled_images: res.require_fully_labeled_images ?? null,
        unlabeled_items_on_exported_images:
          res.unlabeled_items_on_exported_images ?? null,
        images_with_unlabeled_items: res.images_with_unlabeled_items ?? null,
        images_dropped_not_fully_labeled: res.images_dropped_not_fully_labeled ?? null,
        skipped_items: res.skipped_items ?? null,
      };
      if (res.status === 'running' || res.status === 'pending') {
        toastStore.info(`Export started: ${res.status}`);
        startPolling();
        return;
      }
      exportRunning = false;
      if (res.status === 'success') {
        toastStore.success(`Export complete: ${res.export_dir ?? res.status}`);
        // Bug 1: the "frozen multi-class export" card/registry download
        // buttons are gated on `hasMulticlassExport`, which only
        // `loadAll()` (re-fetching `{API_PREFIX}/export/datasets`) can
        // set — reload now instead of waiting for the user to hit Refresh.
        // loadAll() also re-fetches GET /export/status, which fills in
        // class_count/class_split_counts/group_key — fields the
        // synchronous POST response above doesn't carry.
        await loadAll();
      } else if (res.status === 'failed') {
        toastStore.error(`Export failed: ${res.message ?? 'unknown error'}`);
      } else {
        toastStore.info(`Export: ${res.status}`);
      }
    } catch (e) {
      exportRunning = false;
      // 422 "nothing to export: <reason>" (e.g. require_fully_labeled_images
      // dropped every candidate image) surfaces via ApiError's detail text.
      toastStore.error(`Export failed: ${(e as Error).message}`);
    }
  }

  function downloadUrl(url: string, suggestedName: string): void {
    const a = document.createElement('a');
    a.href = url;
    a.download = suggestedName;
    a.target = '_blank';
    a.rel = 'noopener';
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // ---- holdout freeze ----------------------------------------------------

  const totalTestCrops = $derived(holdout?.total ?? 0);
  const testFrozen = $derived(totalTestCrops > 0);
  // Server-flagged deficient classes (`{API_PREFIX}/test_holdout/stats`'s
  // per-bucket `deficient`, falling back to the served `min_test_per_class`
  // when a bucket omits the flag) — never a hardcoded "< 5".
  const deficientClassCount = $derived(rows.filter((r) => r.testDeficient).length);

  function openFreeze(): void {
    freezePercent = 10;
    freezeOpen = true;
  }

  function closeFreeze(): void {
    if (freezeBusy) return;
    freezeOpen = false;
  }

  function closeExportModal(): void {
    if (exportRunning) return;
    exportModalOpen = false;
  }

  async function submitFreeze(): Promise<void> {
    const ok = window.confirm(
      `Freeze ${freezePercent}% of validated crops as the test set? This is ` +
        'one-shot per dataset version (Plan §B4) — re-running requires ?force=true ' +
        'and is recorded in the manifest. Selection is deterministic ' +
        '(SHA1 of each crop id, per class) — no seed to pick.',
    );
    if (!ok) return;
    freezeBusy = true;
    try {
      const res = await freezeTestHoldout({ percent: freezePercent });
      toastStore.success(
        `Frozen: ${res.n_frozen} crops across ${res.n_classes_covered} classes via ` +
          `${res.selection} (min ${res.min_per_class}/class).`,
      );
      freezeOpen = false;
      await loadAll();
    } catch (e) {
      toastStore.error(`Freeze failed: ${(e as Error).message}`);
    } finally {
      freezeBusy = false;
    }
  }

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
  // surface only exposes per_class + ingestion summary; we hit the
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
        hddSourcesError = `Dataset totals unavailable: ${(e as Error).message}`;
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

  <!-- Test holdout status card -->
  <section class="surface p-4">
    <header class="mb-2 flex flex-wrap items-center gap-3">
      <h2 class="text-sm font-semibold text-zinc-300">Test holdout</h2>
      <span class="text-xs text-zinc-500">
        red badge: a class with frozen test crops that the server flags deficient (below {holdout?.min_test_per_class ??
          '…'}). Classes with no validated crops have no test crops and are not flagged.
      </span>
      <span class="grow"></span>
      {#if !testFrozen}
        <button class="btn btn-primary" type="button" onclick={openFreeze}>
          Freeze test set
        </button>
      {:else}
        <span
          class="rounded-md border border-green-500/40 bg-green-500/10 px-2 py-1 text-xs text-green-200"
        >
          frozen — {totalTestCrops.toLocaleString()} crops
        </span>
      {/if}
    </header>
    {#if testFrozen}
      <div class="flex flex-wrap items-center gap-3 text-xs">
        <span class="text-zinc-400">
          {totalTestCrops.toLocaleString()} test crops across
          {(holdout?.by_class?.length ?? 0).toString()} classes.
        </span>
        {#if deficientClassCount > 0}
          <span
            class="rounded-md border border-red-500/40 bg-red-500/10 px-2 py-0.5 text-red-200"
          >
            {deficientClassCount} class{deficientClassCount === 1 ? '' : 'es'} below {holdout?.min_test_per_class ??
              '…'} test crops
          </span>
        {/if}
      </div>
    {:else}
      <p class="text-xs text-zinc-500">
        Test set is not yet frozen. Plan §B4: freezing is one-shot per dataset version.
      </p>
    {/if}
  </section>

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

  {#snippet classRow(row: ExportRow)}
    <tr class="border-b border-zinc-900 hover:bg-zinc-900/40">
      <td class="px-3 py-1.5 text-zinc-200">{row.class_name}</td>
      <td class="px-3 py-1.5 font-mono text-xs text-zinc-500">{row.class_id}</td>
      <td class="px-3 py-1.5 text-right font-mono text-zinc-400">
        {row.total.toLocaleString()}
      </td>
      <td class="px-3 py-1.5 text-right font-mono text-zinc-400">
        {row.validated.toLocaleString()}
      </td>
      <td class="px-3 py-1.5 text-right font-mono text-zinc-200">
        {row.trainable.toLocaleString()}
      </td>
      <td class="px-3 py-1.5 text-right font-mono text-zinc-300">
        {row.aug_target.toLocaleString()}
      </td>
      <td class="px-3 py-1.5 text-right">
        <span
          class="rounded-md border px-1.5 py-0.5 font-mono text-xs {gapClass(
            row.trainableGap,
          )}"
          title={gapCellTitle(row.trainableGap, stats?.thresholds?.block_below ?? null)}
        >
          {row.trainableGap > 0 ? '+' : ''}{row.trainableGap.toLocaleString()}
        </span>
      </td>
      <td class="px-3 py-1.5 text-right">
        <span
          class="rounded-md border px-1.5 py-0.5 font-mono text-xs {testBadge(
            row.testDeficient,
          )}"
        >
          {row.test_count}
        </span>
      </td>
      <td class="px-3 py-1.5 text-right">
        {#if row.adequacy == null}
          <span class="font-mono text-xs text-zinc-500">—</span>
        {:else}
          <span
            class="rounded-md border px-1.5 py-0.5 font-mono text-xs {adequacyClass(
              row.adequacy,
            )}"
          >
            {row.adequacy}
          </span>
        {/if}
      </td>
    </tr>
  {/snippet}

  <!-- Dataset table -->
  <section class="surface max-h-[70vh] shrink-0 overflow-auto">
    {#if loading && rows.length === 0}
      <div class="p-6 text-sm text-zinc-500">Loading dataset stats…</div>
    {:else if error}
      <div class="p-6 text-sm text-red-300">{error}</div>
    {:else if rows.length === 0}
      <div class="p-6 text-sm text-zinc-500">
        No classes yet — ingest some data first.
      </div>
    {:else}
      <table class="w-full text-sm">
        <thead
          class="sticky top-0 z-10 border-b border-zinc-800 bg-zinc-950 text-left text-xs uppercase text-zinc-400"
        >
          <tr>
            <th
              class="cursor-pointer px-3 py-2 font-medium hover:text-zinc-100"
              onclick={() => setSort('class_name')}>Class</th
            >
            <th
              class="cursor-pointer px-3 py-2 font-medium hover:text-zinc-100"
              onclick={() => setSort('class_id')}>ID</th
            >
            <!-- DQ-m9 (docs/design/data-quality-pass-2026-09-24.md): this
                 "Total" is every crop with that class_id
                 (GET {API_PREFIX}/stats/classes) — a DIFFERENT number than the
                 sidebar's / /classes' "Total" (GET {API_PREFIX}/classes'
                 sample_count, the class-cluster bucket size). The two can
                 legitimately disagree by a lot: a region-bound class can
                 count region boxes in the sidebar/classes but show 0 in
                 this column, because no crop's own class_id is the region
                 class — regions are sub-boxes on other items, counted in
                 the cluster bucket but not in the class-id total. -->
            <th
              class="cursor-pointer px-3 py-2 text-right font-medium hover:text-zinc-100"
              onclick={() => setSort('total')}
              title="Crops with this class_id (GET {scoped()}/stats/classes) — not the same as the class-cluster bucket size shown in the sidebar and on /classes."
              >Total (labelled)</th
            >
            <!-- E1 (visual audit 2026-09-24): Validated counts the frozen
                 test crops too; Trainable and Gap are the served
                 `trainable`/`trainable_gap`. -->
            <th
              class="cursor-pointer px-3 py-2 text-right font-medium hover:text-zinc-100"
              onclick={() => setSort('validated')}
              title="Validated crops, including any frozen as test holdout">Validated</th
            >
            <th
              class="cursor-pointer px-3 py-2 text-right font-medium hover:text-zinc-100"
              onclick={() => setSort('trainable')}
              title={TRAINABLE_COLUMN_TITLE}>Trainable</th
            >
            <th
              class="cursor-pointer px-3 py-2 text-right font-medium hover:text-zinc-100"
              onclick={() => setSort('aug_target')}>Aug target</th
            >
            <th
              class="cursor-pointer px-3 py-2 text-right font-medium hover:text-zinc-100"
              onclick={() => setSort('trainableGap')}
              title={GAP_COLUMN_TITLE}>Gap</th
            >
            <th class="px-3 py-2 text-right font-medium" title="Frozen test holdout crops"
              >Test (held out)</th
            >
            <th class="px-3 py-2 text-right font-medium">Adequacy</th>
          </tr>
        </thead>
        <tbody>
          {#each activeRows as row (row.class_id)}
            {@render classRow(row)}
          {/each}
          {#if emptyRows.length > 0}
            <tr class="border-b border-zinc-900">
              <td colspan="9" class="px-3 py-1.5">
                <button
                  type="button"
                  class="text-xs text-zinc-400 hover:text-zinc-200"
                  aria-expanded={showEmptyClasses}
                  data-testid="export-empty-classes-toggle"
                  onclick={() => (showEmptyClasses = !showEmptyClasses)}
                >
                  {showEmptyClasses ? '▾' : '▸'}
                  {emptyRows.length}
                  {emptyRows.length === 1 ? 'class' : 'classes'} with no validated crops
                </button>
              </td>
            </tr>
            {#if showEmptyClasses}
              {#each emptyRows as row (row.class_id)}
                {@render classRow(row)}
              {/each}
            {/if}
          {/if}
        </tbody>
      </table>
    {/if}
  </section>

  <!-- Export controls + downloads + training command -->
  <section class="surface p-4">
    <h2 class="mb-3 text-sm font-semibold text-zinc-300">Export to staging</h2>
    <div class="flex flex-wrap items-end gap-3">
      <label class="text-xs">
        <span class="mb-1 block text-zinc-400">Version tag (optional)</span>
        <input
          type="text"
          bind:value={versionTag}
          placeholder="optional, e.g. baseline-a"
          class="input w-48"
        />
      </label>
      <label
        class="flex cursor-pointer items-center gap-2 pb-1.5 text-xs text-zinc-300"
        title="Drop any exported image that still has an unvalidated or otherwise unlabeled object on it, instead of letting the detector learn that object as background."
      >
        <input
          type="checkbox"
          bind:checked={requireFullyLabeled}
          class="h-4 w-4 cursor-pointer accent-blue-500"
        />
        Only images whose every object is labeled
      </label>
      <button
        type="button"
        class="btn btn-primary"
        onclick={() => void runExport()}
        disabled={exportRunning || nothingExportable}
        title={nothingExportable
          ? 'Nothing to export yet — every class is at 0 validated crops or the served block adequacy tier.'
          : undefined}
      >
        {exportRunning
          ? 'Exporting…'
          : exportState?.status === 'success'
            ? 'Re-export'
            : 'Export'}
      </button>

      <span class="grow"></span>

      <div class="flex flex-wrap items-center gap-2">
        <button
          type="button"
          class="btn"
          onclick={() => downloadUrl(getClassRegistryUrl(), 'class_registry.json')}
          disabled={!hasMulticlassExport}
          title={hasMulticlassExport
            ? ''
            : 'No frozen multi-class (yolo) export on disk yet'}
        >
          class_registry.json
        </button>
        <button
          type="button"
          class="btn"
          onclick={() => downloadUrl(getDataYamlUrl(), 'data.yaml')}
          disabled={!hasMulticlassExport}
          title={hasMulticlassExport
            ? ''
            : 'No frozen multi-class (yolo) export on disk yet'}
        >
          data.yaml
        </button>
        <button
          type="button"
          class="btn"
          onclick={() => downloadUrl(getManifestUrl(), 'manifest.json')}
          disabled={!hasMulticlassExport}
          title={hasMulticlassExport
            ? ''
            : 'No frozen multi-class (yolo) export on disk yet'}
        >
          manifest.json
        </button>
        {#if !hasMulticlassExport}
          <span class="text-[11px] text-zinc-500">No frozen multi-class export yet</span>
        {/if}
      </div>
    </div>

    {#if exportState}
      <div class="mt-3 text-xs text-zinc-400">
        Last status: <span class="font-mono text-zinc-200">{exportState.status}</span>
        {#if exportState.last_run}
          · {new Date(exportState.last_run).toLocaleString()}
        {/if}
        {#if exportState.export_dir}
          · <span class="font-mono text-zinc-300">{exportState.export_dir}</span>
        {/if}
        {#if exportState.error}
          · <span class="text-red-300">{exportState.error}</span>
        {/if}
      </div>
      {#if exportState.status === 'success'}
        <p class="mt-2 text-xs text-zinc-400">
          Next: <a
            href={resolve(projectHref('/train'))}
            class="text-blue-400 underline hover:text-blue-300"
            >train on this export from the Train cockpit</a
          >.
        </p>
      {/if}

      <!-- Image/object counts + per-class table (OpenProcessor 4c9499a's
           `GET {API_PREFIX}/export/status`, ExportStatusResponse — one
           image + one label file per source image, one line per object).
           A null `image_count`/`object_count` renders "—", never 0
           (formatCount); with none of them served (no export yet) this
           whole block doesn't render rather than showing blanks. -->
      {#if exportState.image_count != null || exportState.object_count != null || exportState.class_count != null || exportState.split_counts}
        <div class="mt-3 flex flex-wrap gap-2 text-xs">
          {#if exportState.image_count != null || exportState.object_count != null}
            <span class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1">
              <span class="ml-1 font-mono text-zinc-200"
                >{formatCount(exportState.object_count)}</span
              >
              <span class="text-zinc-500">objects in</span>
              <span class="ml-1 font-mono text-zinc-200"
                >{formatCount(exportState.image_count)}</span
              >
              <span class="text-zinc-500">images</span>
              {#if exportState.group_key}
                <span class="ml-1 text-zinc-500"
                  >(grouped by {exportState.group_key})</span
                >
              {/if}
            </span>
          {/if}
          {#if exportState.class_count != null}
            <!-- E2 (visual audit 2026-09-24): the served class_count is the
                 registry size; say how many classes actually have objects.
                 #36 item 6: prefer the served classes_with_objects over the
                 client-side count from class_split_counts, which is kept
                 only as the fallback for an export/backend written before
                 the field existed. -->
            <span
              class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1"
              data-testid="export-class-count"
            >
              {#if exportState.classes_with_objects != null}
                <span class="font-mono text-zinc-200"
                  >{exportState.classes_with_objects}</span
                >
                <span class="text-zinc-500">classes with objects</span>
                <span class="ml-1 text-zinc-500"
                  >({exportState.class_count} in registry)</span
                >
              {:else if exportState.class_split_counts}
                <span class="font-mono text-zinc-200"
                  >{exportClassSplit.withObjects.length}</span
                >
                <span class="text-zinc-500">classes with objects</span>
                <span class="ml-1 text-zinc-500"
                  >({exportState.class_count} in registry)</span
                >
              {:else}
                <span class="text-zinc-500">classes in registry</span>
                <span class="ml-1 font-mono text-zinc-200">{exportState.class_count}</span
                >
              {/if}
            </span>
          {/if}
          {#if exportState.split_counts}
            <span
              class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1 font-mono"
              title="Images per split"
            >
              images: train {exportState.split_counts.train.toLocaleString()} · val {exportState.split_counts.val.toLocaleString()}
              · test {exportState.split_counts.test.toLocaleString()}
            </span>
          {/if}
          {#if exportState.split_object_counts}
            <span
              class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1 font-mono"
              title="Objects (label lines) per split"
            >
              objects: train {exportState.split_object_counts.train.toLocaleString()} · val
              {exportState.split_object_counts.val.toLocaleString()}
              · test {exportState.split_object_counts.test.toLocaleString()}
            </span>
          {/if}
        </div>
      {/if}

      <!-- Partial-frame policy + counts (4c9499a) — null on an older
           export (formatCount renders "—"); the whole block hides when
           nothing here was ever recorded. -->
      {#if exportState.require_fully_labeled_images != null || exportState.unlabeled_items_on_exported_images != null || exportState.images_with_unlabeled_items != null || exportState.images_dropped_not_fully_labeled != null || exportState.skipped_items != null}
        <div class="mt-2 flex flex-wrap gap-2 text-xs text-zinc-400">
          {#if exportState.require_fully_labeled_images != null}
            <span class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1">
              require_fully_labeled_images: <span class="font-mono text-zinc-200"
                >{exportState.require_fully_labeled_images ? 'true' : 'false'}</span
              >
            </span>
          {/if}
          {#if exportState.unlabeled_items_on_exported_images != null}
            <span
              class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1"
              title="Objects on exported images the export did not label — learned as background."
            >
              unlabeled objects on exported images: <span class="font-mono text-zinc-200"
                >{formatCount(exportState.unlabeled_items_on_exported_images)}</span
              >
            </span>
          {/if}
          {#if exportState.images_with_unlabeled_items != null}
            <span class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1">
              images with an unlabeled object: <span class="font-mono text-zinc-200"
                >{formatCount(exportState.images_with_unlabeled_items)}</span
              >
            </span>
          {/if}
          {#if exportState.images_dropped_not_fully_labeled != null}
            <span
              class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1"
              title="Images left out by require_fully_labeled_images (0 when it was off)."
            >
              images dropped (not fully labeled): <span class="font-mono text-zinc-200"
                >{formatCount(exportState.images_dropped_not_fully_labeled)}</span
              >
            </span>
          {/if}
          {#if exportState.skipped_items != null}
            <span
              class="rounded-md border border-zinc-700 bg-zinc-900/40 px-2 py-1"
              title="Validated items the export couldn't write: no source image id, or no usable box/class."
              data-testid="export-skipped-items"
            >
              skipped items: <span class="font-mono text-zinc-200"
                >{formatCount(exportState.skipped_items.no_image_id)} no image id · {formatCount(
                  exportState.skipped_items.no_usable_box_or_class,
                )} no usable box/class</span
              >
            </span>
          {/if}
        </div>
      {/if}

      {#if exportState.class_split_counts && exportState.class_split_counts.length > 0}
        <details class="mt-3 text-xs">
          <summary class="cursor-pointer text-zinc-400 hover:text-zinc-200">
            Per-class object counts ({exportClassSplit.withObjects.length} classes with objects)
          </summary>
          <div class="mt-2 max-h-64 overflow-auto rounded border border-zinc-800">
            <table class="w-full text-xs">
              <thead
                class="sticky top-0 border-b border-zinc-800 bg-zinc-950 text-left uppercase text-zinc-500"
              >
                <tr>
                  <th class="px-2 py-1 font-medium">Class</th>
                  <th class="px-2 py-1 text-right font-medium">Train (objects)</th>
                  <th class="px-2 py-1 text-right font-medium">Val (objects)</th>
                  <th class="px-2 py-1 text-right font-medium">Test (objects)</th>
                </tr>
              </thead>
              <tbody>
                {#each exportClassSplit.withObjects as c (c.class_id)}
                  {@const missing = c.train === 0 || c.val === 0}
                  <tr
                    class="border-b border-zinc-900 {missing
                      ? 'bg-red-500/10 text-red-200'
                      : 'text-zinc-300'}"
                  >
                    <td class="px-2 py-1">{c.class_name}</td>
                    <td class="px-2 py-1 text-right font-mono"
                      >{c.train.toLocaleString()}</td
                    >
                    <td class="px-2 py-1 text-right font-mono"
                      >{c.val.toLocaleString()}</td
                    >
                    <td class="px-2 py-1 text-right font-mono"
                      >{c.test.toLocaleString()}</td
                    >
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
          {#if exportClassSplit.empty.length > 0}
            <p class="mt-1 text-zinc-500" data-testid="export-empty-classes">
              {exportClassSplit.empty.length} class{exportClassSplit.empty.length === 1
                ? ''
                : 'es'} with no objects in this export
            </p>
          {/if}
        </details>
      {/if}
    {/if}
  </section>
</div>

<!-- Export progress modal -->
{#if exportModalOpen}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Export progress"
    tabindex="-1"
    use:focusOnMount
    use:trapFocus={{ onEscape: closeExportModal }}
    onclick={(e) => {
      if (e.target === e.currentTarget) closeExportModal();
    }}
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-3 text-base font-semibold">YOLO export</h3>
      {#if exportState?.status === 'running' || exportRunning}
        <p class="mb-3 text-xs text-zinc-400">
          Running… polling every 5s. Safe to leave the page open.
        </p>
        {#if exportState?.progress != null}
          <div class="mb-2 h-2 w-full overflow-hidden rounded bg-zinc-900">
            <div
              class="h-full bg-blue-500"
              style:width="{Math.round((exportState.progress ?? 0) * 100)}%"
            ></div>
          </div>
        {/if}
        <p class="text-xs font-mono text-zinc-300">{exportState?.message ?? '…'}</p>
      {:else if exportState?.status === 'success'}
        <p class="mb-2 text-sm text-green-300">Export complete.</p>
        {#if exportState.export_dir}
          <p class="mb-3 break-all font-mono text-xs text-zinc-300">
            {exportState.export_dir}
          </p>
        {/if}
        {#if exportState.split_counts || exportState.image_count != null || exportState.object_count != null}
          <p class="mb-3 font-mono text-xs text-zinc-300">
            {formatCount(exportState.object_count)} objects in {formatCount(
              exportState.image_count,
            )} images
            {exportState.class_count != null
              ? `· ${exportState.class_count} classes`
              : ''}
            {#if exportState.split_counts}
              · images train {exportState.split_counts.train.toLocaleString()} · val {exportState.split_counts.val.toLocaleString()}
              · test {exportState.split_counts.test.toLocaleString()}
            {/if}
          </p>
        {/if}
        {#if exportState.images_dropped_not_fully_labeled}
          <p class="mb-3 text-xs text-orange-300">
            {formatCount(exportState.images_dropped_not_fully_labeled)} image(s) dropped — not
            fully labeled.
          </p>
        {/if}
        <div class="flex flex-wrap gap-2">
          <button
            type="button"
            class="btn"
            onclick={() => downloadUrl(getManifestUrl(), 'manifest.json')}
          >
            Download manifest.json
          </button>
        </div>
      {:else if exportState?.status === 'failed'}
        <p class="mb-2 text-sm text-red-300">Export failed.</p>
        <p class="font-mono text-xs text-red-200">
          {exportState.error ?? 'unknown error'}
        </p>
      {:else}
        <p class="text-xs text-zinc-400">No active export.</p>
      {/if}
      <div class="mt-4 flex justify-end">
        <button
          type="button"
          class="btn"
          onclick={closeExportModal}
          disabled={exportRunning}>Close</button
        >
      </div>
    </div>
  </div>
{/if}

<!-- Freeze test holdout modal -->
{#if freezeOpen}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Freeze test holdout"
    tabindex="-1"
    use:focusOnMount
    use:trapFocus={{ onEscape: closeFreeze }}
    onclick={(e) => {
      if (e.target === e.currentTarget) closeFreeze();
    }}
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-2 text-base font-semibold">Freeze test holdout</h3>
      <div
        class="mb-3 rounded border border-orange-500/40 bg-orange-500/10 px-3 py-2 text-xs text-orange-200"
      >
        One-shot per dataset version. Deterministic per-class selection (SHA1 of each crop
        id) — the same cohort always freezes the same set, so there's no seed to pick
        (Plan §B4).
      </div>
      <label class="mb-3 block text-sm">
        <span class="mb-1 block text-zinc-400">Percent of validated crops</span>
        <input
          type="number"
          min="1"
          max="50"
          bind:value={freezePercent}
          class="input w-full"
        />
      </label>
      <div class="flex justify-end gap-2">
        <button type="button" class="btn" onclick={closeFreeze} disabled={freezeBusy}>
          Cancel
        </button>
        <button
          type="button"
          class="btn btn-primary"
          onclick={() => void submitFreeze()}
          disabled={freezeBusy}
        >
          {freezeBusy ? 'Freezing…' : 'Freeze'}
        </button>
      </div>
    </div>
  </div>
{/if}
