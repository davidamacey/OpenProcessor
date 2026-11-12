// Wire types of the Table view. Pinned to the generated contract by
// `contract/tableContract.test.ts`.
import type { HiddenCounts } from '$lib/types';
import type { SortOption } from '$lib/types_cursor';

export type TableRows = 'detections' | 'images';
export type CellTone =
  'validated' | 'machine' | 'unlabeled' | 'holdout' | 'rejected' | 'muted';

export interface TableColumn {
  id: string;
  label: string;
  sortable: boolean;
  align: 'start' | 'end' | 'center';
  kind: 'thumbnail' | 'text' | 'number' | 'date' | 'badge';
}

/** `GET /table/columns`. */
export interface TableColumns {
  rows: TableRows;
  columns: TableColumn[];
  /** The column ids shown when `columns` is not given. */
  default: string[];
}

export interface TableCell {
  /** Already formatted; shown verbatim. */
  text: string;
  /** The raw wire value the text was made from. */
  value?: string | number | boolean | null;
  tone?: CellTone | null;
}

export interface TableRow {
  /** The item of a detection row; null on an image row. */
  crop_id: string | null;
  image_id: string;
  thumbnail_url: string;
  cells: Record<string, TableCell>;
}

/** `GET /table`. */
export interface TablePage {
  rows: TableRows;
  columns: TableColumn[];
  items: TableRow[];
  next_cursor?: string | null;
  total: number;
  limit: number;
  sort_applied: string;
  sort_options: SortOption[];
  hidden: HiddenCounts;
  join_truncated: boolean;
  empty_reason?: 'no_items' | 'no_match' | null;
  empty_message?: string | null;
}

/** What the table asks for beside the shared item filter. */
export interface TableQuery {
  rows: TableRows;
  /** Column ids in order; the served default when empty. */
  columns: string[];
  sort: string | null;
  cursor: string | null;
  limit?: number;
}
