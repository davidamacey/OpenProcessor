<script lang="ts" module>
  export interface Column<R> {
    id: string;
    label: string;
    /** The sort key; a column without it is not sortable. */
    value?: (row: R) => number | string;
    align?: 'left' | 'right';
    title?: string;
  }
</script>

<script lang="ts" generics="T">
  /**
   * A compact table for many rows: a sticky header inside a scrolling frame, click-to-sort
   * headers (`aria-sort`, ascending then descending), and one `cell` snippet that renders each
   * body cell. Sorting is by the column's `value(row)`; filtering and what a row is are the
   * caller's. The frame scrolls both ways, so a wide matrix never widens the page.
   */
  import { untrack, type Snippet } from 'svelte';
  import Icon from './Icon.svelte';

  interface Props {
    columns: Column<T>[];
    rows: T[];
    rowKey: (row: T) => string;
    label: string;
    cell: Snippet<[T, Column<T>]>;
    /** Height cap of the scrolling frame, a Tailwind class. */
    maxHeightClass?: string;
    initialSort?: { column: string; dir: 'asc' | 'desc' };
  }
  let {
    columns,
    rows,
    rowKey,
    label,
    cell,
    maxHeightClass = 'max-h-[60vh]',
    initialSort,
  }: Props = $props();

  // The starting sort only; the user's clicks own it afterwards.
  let sort = $state<{ column: string; dir: 'asc' | 'desc' } | null>(
    untrack(() => initialSort) ?? null,
  );

  const sorted = $derived.by(() => {
    const col = columns.find((c) => c.id === sort?.column);
    if (!col?.value || !sort) return rows;
    const value = col.value;
    const sign = sort.dir === 'asc' ? 1 : -1;
    return [...rows].sort((a, b) => {
      const x = value(a);
      const y = value(b);
      const cmp =
        typeof x === 'number' && typeof y === 'number'
          ? x - y
          : String(x).localeCompare(String(y));
      return cmp * sign;
    });
  });

  function toggle(col: Column<T>): void {
    if (!col.value) return;
    sort =
      sort?.column === col.id
        ? { column: col.id, dir: sort.dir === 'asc' ? 'desc' : 'asc' }
        : { column: col.id, dir: col.align === 'right' ? 'desc' : 'asc' };
  }
  const ariaSort = (col: Column<T>) =>
    sort?.column === col.id
      ? sort.dir === 'asc'
        ? 'ascending'
        : 'descending'
      : col.value
        ? 'none'
        : undefined;
</script>

<div
  class="overflow-auto rounded-card border border-line {maxHeightClass}"
  data-testid="data-table"
>
  <table class="w-full border-collapse text-body" aria-label={label}>
    <thead class="sticky top-0 z-10 bg-surface-2 text-meta text-fg-2">
      <tr>
        {#each columns as col (col.id)}
          <th
            scope="col"
            class="border-b border-line px-3 py-1.5 font-medium whitespace-nowrap {col.align ===
            'right'
              ? 'text-right'
              : 'text-left'}"
            aria-sort={ariaSort(col)}
            title={col.title}
          >
            {#if col.value}
              <button
                type="button"
                class="inline-flex items-center gap-1 hover:text-fg-1"
                onclick={() => toggle(col)}
              >
                {col.label}
                {#if sort?.column === col.id}
                  <Icon name={sort.dir === 'asc' ? 'arrow-up' : 'arrow-down'} size={12} />
                {/if}
              </button>
            {:else}
              {col.label}
            {/if}
          </th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#each sorted as row (rowKey(row))}
        <tr class="border-b border-line last:border-b-0 hover:bg-surface-2">
          {#each columns as col (col.id)}
            <td
              class="px-3 py-1 {col.align === 'right' ? 'text-right tabular-nums' : ''}"
            >
              {@render cell(row, col)}
            </td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
</div>
