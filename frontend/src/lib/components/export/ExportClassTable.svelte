<script lang="ts">
  /** /export's per-class dataset table: sortable, with the classes that
   *  have no validated or held-out crop folded behind one toggle row. */
  import {
    GAP_COLUMN_TITLE,
    gapCellTitle,
    TRAINABLE_COLUMN_TITLE,
    type ExportRow,
  } from '$lib/export/exportDatasetRows';
  import { scoped } from '$lib/api';

  interface Props {
    rows: ExportRow[];
    loading: boolean;
    error: string | null;
    /** The served `thresholds.block_below` the Gap cell titles cite. */
    blockBelow: number | null;
  }

  let { rows, loading, error, blockBelow }: Props = $props();

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

  const sortedRows = $derived.by((): ExportRow[] => {
    const list = [...rows];
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
  const activeRows = $derived(
    sortedRows.filter((r) => r.validated > 0 || r.test_count > 0),
  );
  const emptyRows = $derived(
    sortedRows.filter((r) => !(r.validated > 0 || r.test_count > 0)),
  );
  let showEmptyClasses = $state(false);

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
</script>

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
        title={gapCellTitle(row.trainableGap, blockBelow)}
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
    <div class="p-6 text-sm text-zinc-500">No classes yet — ingest some data first.</div>
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
