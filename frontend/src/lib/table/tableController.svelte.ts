/**
 * The Table view's state. The server sorts, pages and formats; this controller sends the
 * choice (rows, columns, a served sort id) as query parameters, mirrors it into the URL and
 * renders the served rows as they come. Selection is by row key (`crop_id`, else `image_id`).
 */
import { SvelteSet } from 'svelte/reactivity';
import { getTable, getTableColumns } from '$lib/api_table';
import { createPager } from '$lib/pager.svelte';
import type { HiddenCounts } from '$lib/types';
import type { NearDupFilter } from '$lib/types_images';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import type { SortOption } from '$lib/types_cursor';
import type {
  TableColumn,
  TableColumns,
  TablePage,
  TableRow,
  TableRows,
} from '$lib/types_table';

/** What the page URL says about the table. */
export interface TableUrlState {
  rows: TableRows;
  /** Empty means the served default. */
  columns: string[];
  sort: string | null;
}

export interface TableDeps {
  /** What the page was opened with (read once). */
  readonly initial: TableUrlState;
  itemFilter(): ItemFilterQuery;
  /** The near-duplicate role from the URL; null lists every row. */
  nearDup(): NearDupFilter | null;
  writeUrl(state: TableUrlState): void;
}

export const rowKey = (r: TableRow): string => r.crop_id ?? r.image_id;

export function createTableController(deps: TableDeps) {
  const { initial } = deps;
  let rowsKind = $state<TableRows>(initial.rows);
  let chosen = $state<string[]>([...initial.columns]);
  let sort = $state<string | null>(initial.sort);
  let vocabulary = $state<TableColumns | null>(null);
  let shownColumns = $state<TableColumn[]>([]);
  let sortApplied = $state('');
  let sortOptions = $state<SortOption[]>([]);
  let emptyMessage = $state<string | null>(null);
  let hidden = $state<HiddenCounts>({ test_holdout: 0, excluded: 0 });
  const selected = new SvelteSet<string>();
  let focused = $state<string | null>(null);

  const pager = createPager<TableRow>({
    fetchPage: async (cursor) => {
      const res: TablePage = await getTable(
        { rows: rowsKind, columns: chosen, sort, cursor, near_dup: deps.nearDup() },
        deps.itemFilter(),
      );
      shownColumns = res.columns;
      sortApplied = res.sort_applied;
      sortOptions = res.sort_options;
      emptyMessage = res.empty_message ?? null;
      hidden = res.hidden;
      return res;
    },
    keyOf: rowKey,
  });

  function writeUrl(): void {
    deps.writeUrl({ rows: rowsKind, columns: chosen, sort });
  }

  async function reload(): Promise<void> {
    selected.clear();
    focused = null;
    writeUrl();
    await pager.loadFirst();
  }

  /** The served option for a column: the other direction when this column is applied. */
  function nextSortId(columnId: string): string | null {
    const ids = sortOptions
      .map((o) => o.id)
      .filter((id) => id.startsWith(`${columnId}:`));
    if (ids.length === 0) return null;
    const applied = ids.indexOf(sortApplied);
    return ids[(applied + 1) % ids.length];
  }

  return {
    get rowsKind() {
      return rowsKind;
    },
    get vocabulary() {
      return vocabulary;
    },
    get columns() {
      return shownColumns;
    },
    get chosenIds() {
      return chosen.length > 0 ? chosen : (vocabulary?.default ?? []);
    },
    get rows() {
      return pager.items;
    },
    get total() {
      return pager.total;
    },
    get loading() {
      return pager.loading;
    },
    get loadingMore() {
      return pager.loadingMore;
    },
    get hasMore() {
      return pager.hasMore;
    },
    get error() {
      return pager.error;
    },
    get sortApplied() {
      return sortApplied;
    },
    get sortOptions() {
      return sortOptions;
    },
    get emptyMessage() {
      return emptyMessage;
    },
    get hidden() {
      return hidden;
    },
    get selectedIds() {
      return selected;
    },
    get focusedId() {
      return focused;
    },

    async load(): Promise<void> {
      vocabulary = await getTableColumns(rowsKind);
      await reload();
    },
    loadMore: () => pager.loadMore(),
    reload,

    async setRows(next: TableRows): Promise<void> {
      if (next === rowsKind) return;
      rowsKind = next;
      chosen = [];
      sort = null;
      vocabulary = await getTableColumns(next);
      await reload();
    },
    async setColumns(ids: string[]): Promise<void> {
      chosen = ids;
      await reload();
    },
    /** A header click: ask for the served sort of the column. Unsortable columns do nothing. */
    async sortBy(columnId: string): Promise<void> {
      const id = nextSortId(columnId);
      if (id === null) return;
      sort = id;
      await reload();
    },

    focus(id: string | null): void {
      focused = id;
    },
    select(ids: string[], on: boolean): void {
      for (const id of ids) {
        if (on) selected.add(id);
        else selected.delete(id);
      }
    },
    clearSelection(): void {
      selected.clear();
    },
  };
}

export type TableController = ReturnType<typeof createTableController>;
