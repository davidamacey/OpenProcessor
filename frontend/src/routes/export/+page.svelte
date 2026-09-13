<script lang="ts">
  import {
    apiBase,
    exportStatus,
    exportYolo,
    freezeTestHoldout,
    getClassRegistryUrl,
    getDataYamlUrl,
    getManifestUrl,
    getStats,
    getTestHoldoutStats,
  } from '$lib/api';
  import type { OpExportStatus, OpStats, OpTestHoldoutStats } from '$lib/types';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('export');
  });

  // ---- data --------------------------------------------------------------

  let stats = $state<OpStats | null>(null);
  let holdout = $state<OpTestHoldoutStats | null>(null);
  let loading = $state<boolean>(false);
  let error = $state<string | null>(null);

  // Sort
  type SortKey = 'class_name' | 'class_id' | 'total' | 'validated' | 'aug_target' | 'gap';
  let sortKey = $state<SortKey>('gap');
  let sortDir = $state<'asc' | 'desc'>('desc');

  // Export
  let versionTag = $state<string>('');
  let exportRunning = $state<boolean>(false);
  let exportState = $state<OpExportStatus | null>(null);
  let pollHandle: ReturnType<typeof setInterval> | null = null;
  let exportModalOpen = $state<boolean>(false);

  // Test holdout freeze
  let freezeOpen = $state<boolean>(false);
  let freezePercent = $state<number>(10);
  let freezeSeed = $state<number>(42);
  let freezeBusy = $state<boolean>(false);

  async function loadAll(): Promise<void> {
    loading = true;
    error = null;
    try {
      const [s, h, e] = await Promise.allSettled([
        getStats(),
        getTestHoldoutStats(),
        exportStatus(),
      ]);
      stats = s.status === 'fulfilled' ? s.value : null;
      holdout = h.status === 'fulfilled' ? h.value : null;
      exportState = e.status === 'fulfilled' ? e.value : null;
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

  // YOLO export targets per class. Plan Section L13: aug_target =
  // clamp(validated, 500, 3000).
  function augTarget(validated: number): number {
    return Math.min(3000, Math.max(500, validated));
  }

  interface Row {
    class_id: number;
    class_name: string;
    total: number;
    validated: number;
    aug_target: number;
    gap: number;
    test_count: number;
  }

  const rows = $derived.by((): Row[] => {
    const stat = stats;
    if (!stat?.per_class) return [];
    const testMap = new Map<number, number>();
    for (const b of holdout?.by_class ?? []) {
      testMap.set(b.key, b.doc_count);
    }
    const list: Row[] = stat.per_class.map((c) => {
      const validated = c.validated_count ?? 0;
      const target = augTarget(validated);
      return {
        class_id: c.class_id,
        class_name: c.class_name,
        total: c.count ?? 0,
        validated,
        aug_target: target,
        gap: target - validated,
        test_count: testMap.get(c.class_id) ?? 0,
      };
    });
    list.sort((a, b) => {
      const dir = sortDir === 'asc' ? 1 : -1;
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir;
      return String(av).localeCompare(String(bv)) * dir;
    });
    return list;
  });

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

  function testBadge(count: number): string {
    if (count >= 5) return 'bg-zinc-800 text-zinc-300 border-zinc-700';
    return 'bg-red-500/20 text-red-200 border-red-500/40';
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
          } else if (s.status === 'failed') {
            toastStore.error(`Export failed: ${s.error ?? 'unknown'}`);
          }
        }
      } catch (e) {
        toastStore.warn(`Status poll failed: ${(e as Error).message}`);
      }
    }, 5000);
  }

  async function runExport(): Promise<void> {
    exportRunning = true;
    exportModalOpen = true;
    try {
      const res = await exportYolo({ version_tag: versionTag.trim() || undefined });
      toastStore.info(`Export started: ${res.job_id ?? res.status}`);
      exportState = {
        status: 'running',
        last_run: new Date().toISOString(),
        job_id: res.job_id,
      };
      startPolling();
    } catch (e) {
      exportRunning = false;
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
  const testDeficient = $derived(rows.filter((r) => r.test_count < 5).length);

  function openFreeze(): void {
    freezePercent = 10;
    freezeSeed = 42;
    freezeOpen = true;
  }

  async function submitFreeze(): Promise<void> {
    const ok = window.confirm(
      `Freeze ${freezePercent}% of validated crops as the test set? This is ` +
        'one-shot per dataset version (Plan §B4) — re-running requires ?force=true ' +
        'and is recorded in the manifest.',
    );
    if (!ok) return;
    freezeBusy = true;
    try {
      const res = await freezeTestHoldout({ percent: freezePercent, seed: freezeSeed });
      toastStore.success(
        `Frozen: ${res.n_frozen} crops across ${res.n_classes_covered} classes.`,
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
  // Pull from the same /curation/stats/dataset response — the existing `getStats`
  // surface only exposes per_class + ingestion summary; we hit the
  // dataset-stats endpoint directly via fetch for the by_source bucket.
  // The thin /curation/stats/dataset response carries an `by_source` HDD bucket
  // array which `getStats` (typed to OpStats) doesn't surface. Hit it
  // directly so we can render the source-distribution chip row.
  $effect(() => {
    void (async () => {
      try {
        const res = await fetch(`${apiBase}/curation/stats/dataset`, { method: 'GET' });
        if (!res.ok) return;
        const ct = res.headers.get('content-type') ?? '';
        if (!ct.includes('application/json')) return;
        const json = (await res.json()) as { by_source?: HddBucket[] };
        if (Array.isArray(json.by_source)) hddSources = json.by_source;
      } catch {
        /* ignore */
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
        red badge if &lt;5 test crops in any class
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
        {#if testDeficient > 0}
          <span
            class="rounded-md border border-red-500/40 bg-red-500/10 px-2 py-0.5 text-red-200"
          >
            {testDeficient} class{testDeficient === 1 ? '' : 'es'} below 5 test crops
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
  {#if hddSources.length > 0}
    <section class="surface p-4">
      <h2 class="mb-2 text-sm font-semibold text-zinc-300">HDD source distribution</h2>
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

  <!-- Dataset table -->
  <section class="surface min-h-0 flex-1 overflow-auto">
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
            <th
              class="cursor-pointer px-3 py-2 text-right font-medium hover:text-zinc-100"
              onclick={() => setSort('total')}>Total</th
            >
            <th
              class="cursor-pointer px-3 py-2 text-right font-medium hover:text-zinc-100"
              onclick={() => setSort('validated')}>Validated</th
            >
            <th
              class="cursor-pointer px-3 py-2 text-right font-medium hover:text-zinc-100"
              onclick={() => setSort('aug_target')}>Aug target</th
            >
            <th
              class="cursor-pointer px-3 py-2 text-right font-medium hover:text-zinc-100"
              onclick={() => setSort('gap')}>Gap</th
            >
            <th class="px-3 py-2 text-right font-medium">Test</th>
          </tr>
        </thead>
        <tbody>
          {#each rows as row (row.class_id)}
            <tr class="border-b border-zinc-900 hover:bg-zinc-900/40">
              <td class="px-3 py-1.5 text-zinc-200">{row.class_name}</td>
              <td class="px-3 py-1.5 font-mono text-xs text-zinc-500">{row.class_id}</td>
              <td class="px-3 py-1.5 text-right font-mono text-zinc-400">
                {row.total.toLocaleString()}
              </td>
              <td class="px-3 py-1.5 text-right font-mono text-zinc-200">
                {row.validated.toLocaleString()}
              </td>
              <td class="px-3 py-1.5 text-right font-mono text-zinc-300">
                {row.aug_target.toLocaleString()}
              </td>
              <td class="px-3 py-1.5 text-right">
                <span
                  class="rounded-md border px-1.5 py-0.5 font-mono text-xs {gapClass(
                    row.gap,
                  )}"
                  title={row.gap <= 0 ? 'on target' : `${row.gap} more crops needed`}
                >
                  {row.gap > 0 ? '+' : ''}{row.gap.toLocaleString()}
                </span>
              </td>
              <td class="px-3 py-1.5 text-right">
                <span
                  class="rounded-md border px-1.5 py-0.5 font-mono text-xs {testBadge(
                    row.test_count,
                  )}"
                >
                  {row.test_count}
                </span>
              </td>
            </tr>
          {/each}
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
          placeholder="e.g. v7.1.0"
          class="input w-48"
        />
      </label>
      <button
        type="button"
        class="btn btn-primary"
        onclick={() => void runExport()}
        disabled={exportRunning}
      >
        {exportRunning
          ? 'Exporting…'
          : exportState?.status === 'success'
            ? 'Re-export'
            : 'Export'}
      </button>

      <span class="grow"></span>

      <div class="flex flex-wrap gap-2">
        <button
          type="button"
          class="btn"
          onclick={() => downloadUrl(getClassRegistryUrl(), 'class_registry.json')}
          disabled={exportState?.status !== 'success'}
        >
          class_registry.json
        </button>
        <button
          type="button"
          class="btn"
          onclick={() => downloadUrl(getDataYamlUrl(), 'data.yaml')}
          disabled={exportState?.status !== 'success'}
        >
          data.yaml
        </button>
        <button
          type="button"
          class="btn"
          onclick={() => downloadUrl(getManifestUrl(), 'manifest.json')}
          disabled={exportState?.status !== 'success'}
        >
          manifest.json
        </button>
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
          Next: <a href="/train" class="text-blue-400 underline hover:text-blue-300"
            >train on this export from the Train cockpit</a
          >.
        </p>
      {/if}
    {/if}
  </section>
</div>

<!-- Export progress modal -->
{#if exportModalOpen}
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Export progress"
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
        <button type="button" class="btn" onclick={() => (exportModalOpen = false)}
          >Close</button
        >
      </div>
    </div>
  </div>
{/if}

<!-- Freeze test holdout modal -->
{#if freezeOpen}
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Freeze test holdout"
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-2 text-base font-semibold">Freeze test holdout</h3>
      <div
        class="mb-3 rounded border border-orange-500/40 bg-orange-500/10 px-3 py-2 text-xs text-orange-200"
      >
        One-shot per dataset version. Stratified by (class × hdd_source) using a fixed
        seed for reproducibility (Plan §B4).
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
      <label class="mb-3 block text-sm">
        <span class="mb-1 block text-zinc-400">Seed</span>
        <input
          type="number"
          bind:value={freezeSeed}
          class="input w-full"
        />
      </label>
      <div class="flex justify-end gap-2">
        <button
          type="button"
          class="btn"
          onclick={() => (freezeOpen = false)}
          disabled={freezeBusy}
        >
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
