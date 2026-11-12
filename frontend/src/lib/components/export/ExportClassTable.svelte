<script lang="ts">
  /** /export's per-class dataset table: sortable, with the classes that
   *  have no validated or held-out crop folded behind one toggle row. */
  import {
    GAP_COLUMN_TITLE,
    gapCellTitle,
    isServedSortField,
    nextClassSort,
    TRAINABLE_COLUMN_TITLE,
    type ExportRow,
  } from '$lib/export/exportDatasetRows';
  import { scoped } from '$lib/api';
  import type { SortOption } from '$lib/types_cursor';

  interface Props {
    rows: ExportRow[];
    loading: boolean;
    error: string | null;
    /** The served `thresholds.block_below` the Gap cell titles cite. */
    blockBelow: number | null;
    /** The served order of `rows` (`<field>:<direction>`) and the orders the server accepts. */
    sortApplied: string | null;
    sortOptions: SortOption[];
    onSort: (sort: string) => void;
  }

  let { rows, loading, error, blockBelow, sortApplied, sortOptions, onSort }: Props =
    $props();

  // The server orders the rows (`GET /stats/classes?sort=`); a header click asks it again.
  const activeRows = $derived(rows.filter((r) => r.validated > 0 || r.test_count > 0));
  const emptyRows = $derived(rows.filter((r) => !(r.validated > 0 || r.test_count > 0)));
  let showEmptyClasses = $state(false);
  const appliedField = $derived(sortApplied?.split(':')[0] ?? null);
  const appliedDirection = $derived(sortApplied?.split(':')[1] ?? null);

  function sortable(field: string): boolean {
    return isServedSortField(field, sortOptions);
  }

  function setSort(field: string): void {
    if (sortable(field)) onSort(nextClassSort(field, sortApplied));
  }

  function ariaSort(field: string): 'ascending' | 'descending' | 'none' {
    if (appliedField !== field) return 'none';
    return appliedDirection === 'asc' ? 'ascending' : 'descending';
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

{#snippet header(field: string, label: string, align: string, title?: string)}
  <th
    class="px-3 py-2 font-medium {align} {sortable(field)
      ? 'cursor-pointer hover:text-zinc-100'
      : ''}"
    aria-sort={ariaSort(field)}
    {title}
    onclick={() => setSort(field)}
  >
    {label}{appliedField === field ? (appliedDirection === 'asc' ? ' ▲' : ' ▼') : ''}
  </th>
{/snippet}

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
    <div class="p-6 text-sm text-zinc-500">
      No classes in this project yet. Add them on the Classes page.
    </div>
  {:else}
    <table class="w-full text-sm">
      <thead
        class="sticky top-0 z-10 border-b border-zinc-800 bg-zinc-950 text-left text-xs uppercase text-zinc-400"
      >
        <tr>
          {@render header('class_name', 'Class', '')}
          {@render header('class_id', 'ID', '')}
          <!-- DQ-m9: this "Total" is every crop with that class_id
               (GET {API_PREFIX}/stats/classes) — a DIFFERENT number than the
               sidebar's / /classes' "Total" (GET {API_PREFIX}/classes'
               sample_count, the class-cluster bucket size). The two can
               legitimately disagree by a lot: a region-bound class can
               count region boxes in the sidebar/classes but show 0 in
               this column, because no crop's own class_id is the region
               class — regions are sub-boxes on other items, counted in
               the cluster bucket but not in the class-id total. -->
          {@render header(
            'count',
            'Total (labelled)',
            'text-right',
            `Crops with this class_id (GET ${scoped()}/stats/classes) — not the same as the class-cluster bucket size shown in the sidebar and on /classes.`,
          )}
          <!-- E1 (visual audit 2026-09-24): Validated counts the frozen
               test crops too; Trainable and Gap are the served
               `trainable`/`trainable_gap`. -->
          {@render header(
            'validated_count',
            'Validated',
            'text-right',
            'Validated crops, including any frozen as test holdout',
          )}
          {@render header('trainable', 'Trainable', 'text-right', TRAINABLE_COLUMN_TITLE)}
          {@render header('aug_target', 'Aug target', 'text-right')}
          {@render header('trainable_gap', 'Gap', 'text-right', GAP_COLUMN_TITLE)}
          {@render header(
            'test_count',
            'Test (held out)',
            'text-right',
            'Frozen test holdout crops',
          )}
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
