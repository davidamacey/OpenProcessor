import type { HiddenCounts } from '$lib/types';
import type { ReprocessRequest } from '$lib/types_import';

/** `SortOption`: one order a list route serves, from its `sort_options`. */
export interface SortOption {
  id: string;
  label: string;
}

/**
 * One page of a cursored list (`GET /crops`, `GET /review/{tab}`, `GET /regions`,
 * `GET /clusters`): the items, the total, and the cursor of the next page
 * (null on the last). The optional keys are what the route serves beside the
 * page; each is read as served, never computed here.
 */
export interface CursorPage<T> {
  items: T[];
  next_cursor: string | null;
  total: number;
  limit: number;
  /** The sort id the server applied, the ids it offers, and why it fell back
   *  from the one asked for (a sort whose field no item carries yet). */
  sort_applied?: string | null;
  sort_options?: SortOption[];
  sort_fallback_reason?: string | null;
  /** Why a zero-result list is empty, with the served sentence. */
  empty_reason?: string | null;
  empty_message?: string | null;
  /** What this page hides: the test holdout and ignored items. */
  hidden?: HiddenCounts | null;
  /** A computed order (`GET /crops?order=`): which method ranked the page and
   *  the pool it drew from. */
  order_method?: string | null;
  order_version?: string | null;
  n_pool?: number | null;
  /** Matching items with no vector, and the request that would embed them. */
  n_unembedded?: number | null;
  suggested_reprocess?: ReprocessRequest | null;
}

/** Where an item sits in a list, and the cursors around it
 *  (`GET /crops/locate`, `GET /review/{tab}/locate`). */
export interface ListPosition {
  /** 0-based index in the whole list; null when the item is not in it. */
  index: number | null;
  total: number | null;
  /** Starts a page at the item; null for the first row. */
  cursor_before: string | null;
  /** Resumes right after the item. */
  cursor_after: string | null;
  reason: string | null;
}
