/**
 * Table wrappers: the columns a table can show and a cursored page of rows. Import from
 * `$lib/api_table` directly; types live in `$lib/types_table`. The sort is a served
 * `sort_options` id, sent as-is.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import type { TableColumns, TablePage, TableQuery, TableRows } from '$lib/types_table';

/** `GET /table/columns`. */
export function getTableColumns(
  rows: TableRows,
  signal?: AbortSignal,
): Promise<TableColumns> {
  return apiFetch<TableColumns>(`${scoped()}/table/columns${qs({ rows })}`, {}, signal);
}

/** `GET /table`: 400 `cursor_invalid` on a stale cursor, 422 `unknown_column`. */
export function getTable(
  query: TableQuery,
  filter: ItemFilterQuery,
  signal?: AbortSignal,
): Promise<TablePage> {
  return apiFetch<TablePage>(
    `${scoped()}/table${qs({
      ...filter,
      rows: query.rows,
      columns: query.columns.length > 0 ? query.columns.join(',') : undefined,
      sort: query.sort ?? undefined,
      cursor: query.cursor ?? undefined,
      limit: query.limit,
    })}`,
    {},
    signal,
  );
}
