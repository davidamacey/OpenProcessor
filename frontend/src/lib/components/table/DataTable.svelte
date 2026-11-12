<!--
  The Table view's grid: a real <table> that renders the served columns and cells verbatim.
  A header click asks the parent for a served sort (it never orders rows here); `aria-sort`
  comes from the served `sort_applied`. The table scrolls inside its own container.
-->
<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  import { resolveApiUrl } from '$lib/api';
  import type { SortOption } from '$lib/types_cursor';
  import type { CellTone, TableColumn, TableRow } from '$lib/types_table';

  interface Props {
    columns: TableColumn[];
    rows: TableRow[];
    sortApplied: string;
    sortOptions: SortOption[];
    selectedIds: ReadonlySet<string>;
    focusedId: string | null;
    onsort: (columnId: string) => void;
    onselect: (key: string, range: boolean) => void;
    onfocus: (key: string) => void;
    onopen: (row: TableRow) => void;
  }

  let {
    columns,
    rows,
    sortApplied,
    sortOptions,
    selectedIds,
    focusedId,
    onsort,
    onselect,
    onfocus,
    onopen,
  }: Props = $props();

  const TONE: Record<CellTone, string> = {
    validated: 'text-success-text',
    machine: 'text-info-text',
    unlabeled: 'text-warn-text',
    holdout: 'text-holdout-text',
    rejected: 'text-danger-text',
    muted: 'text-fg-3',
  };
  const ALIGN = { start: 'text-left', end: 'text-right', center: 'text-center' } as const;

  const keyOf = (r: TableRow): string => r.crop_id ?? r.image_id;
  const appliedColumn = $derived(sortApplied.split(':')[0]);
  const appliedDir = $derived(sortApplied.split(':')[1]);

  function ariaSort(c: TableColumn): 'ascending' | 'descending' | 'none' | undefined {
    if (!c.sortable) return undefined;
    if (c.id !== appliedColumn) return 'none';
    return appliedDir === 'asc' ? 'ascending' : 'descending';
  }

  function sortLabel(c: TableColumn): string {
    const served = sortOptions.filter((o) => o.id.startsWith(`${c.id}:`));
    return served.length > 0 ? `Sort by ${c.label}` : c.label;
  }
</script>

<div
  class="max-h-full overflow-auto rounded border border-line"
  data-testid="data-table-scroll"
>
  <table class="w-full border-collapse text-xs" data-testid="data-table">
    <thead class="sticky top-0 z-10 bg-surface-1">
      <tr>
        <th scope="col" class="w-8 px-2 py-1"><span class="sr-only">Select</span></th>
        {#each columns as c (c.id)}
          <th
            scope="col"
            class="px-2 py-1 font-medium whitespace-nowrap text-fg-2 {c.kind ===
            'thumbnail'
              ? 'w-14'
              : ''} {ALIGN[c.align]}"
            aria-sort={ariaSort(c)}
          >
            {#if c.kind === 'thumbnail' && !c.label}
              <span class="sr-only">Preview</span>
            {:else if c.sortable}
              <button
                type="button"
                class="hover:text-fg-1 focus-visible:outline-2"
                aria-label={sortLabel(c)}
                data-testid="data-table-sort-{c.id}"
                onclick={() => onsort(c.id)}
              >
                {c.label}
                {#if c.id === appliedColumn}
                  <Icon
                    name={appliedDir === 'asc' ? 'chevron-up' : 'chevron-down'}
                    size={12}
                  />
                {/if}
              </button>
            {:else}
              {c.label}
            {/if}
          </th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#each rows as r (keyOf(r))}
        {@const key = keyOf(r)}
        <tr
          class="border-t border-line hover:bg-surface-1/60 {selectedIds.has(key)
            ? 'bg-info-soft'
            : ''} {focusedId === key ? 'outline-2 -outline-offset-2 outline-info' : ''}"
          aria-selected={selectedIds.has(key)}
          data-testid="data-table-row"
          data-row-key={key}
          onclick={() => onfocus(key)}
          ondblclick={() => onopen(r)}
        >
          <td class="px-2 py-1">
            <input
              type="checkbox"
              checked={selectedIds.has(key)}
              aria-label="Select row {key}"
              onclick={(e) => {
                e.stopPropagation();
                onselect(key, e.shiftKey);
              }}
            />
          </td>
          {#each columns as c (c.id)}
            <td class="px-2 py-1 whitespace-nowrap {ALIGN[c.align]}">
              {#if c.kind === 'thumbnail'}
                <img
                  src={resolveApiUrl(r.thumbnail_url)}
                  alt=""
                  loading="lazy"
                  class="h-8 w-8 rounded border border-line object-cover"
                />
              {:else}
                {@const cell = r.cells[c.id]}
                <span
                  class="{c.kind === 'number' ? 'tabular-nums' : ''} {cell?.tone
                    ? TONE[cell.tone]
                    : 'text-fg-1'}">{cell?.text ?? ''}</span
                >
              {/if}
            </td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
</div>
