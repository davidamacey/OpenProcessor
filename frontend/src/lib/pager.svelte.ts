import { apiErrorText } from '$lib/api_errors';
/**
 * Accumulating cursor loader shared by the infinite-scroll grids.
 *
 * Every grid on this app repeats the same shape: fetch the first page into a
 * fresh buffer, then append later pages by the served `next_cursor`, deduping
 * by id because the server collection shifts underneath us as crops get
 * relabeled. Hand-rolling that per page is how `loadMore` on /clusters ended
 * up dropping the `max_rank` / `min_blur_ratio` filters that `loadFirst` sent;
 * one `fetchPage(cursor)` closure per pager makes that class of drift
 * structurally impossible. No view ever sees a page number: the buffer's
 * place in the whole list is the served index of its first item.
 *
 * `items` and `total` stay writable: the pages mutate them optimistically
 * (label, discard, move) between fetches.
 */

export interface PageResult<T> {
  items: T[];
  total?: number | null;
  /** The cursor of the next page; null or absent on the last page. */
  next_cursor?: string | null;
}

export interface Pager<T> {
  items: T[];
  total: number;
  error: string | null;
  readonly loading: boolean;
  readonly loadingMore: boolean;
  /** True once a page has been applied to the buffer since the last reset. */
  readonly loaded: boolean;
  /** The served index of `items[0]` in the whole list: 0 after `loadFirst()`,
   *  the located index after `loadAt()`. */
  readonly firstIndex: number;
  readonly hasMore: boolean;
  /** Reset and fetch the first page. */
  loadFirst(): Promise<void>;
  /** Append the next page; no-op while a fetch is in flight or at the end. */
  loadMore(): Promise<void>;
  /**
   * Reset the buffer and fetch the page that starts at `cursor`, skipping
   * everything before it. For a deep link that resolves to an item deep in a
   * large list (rank 3000), `loadMore()` in a loop would mean one request
   * per intervening page; this issues one. `index` is the served index of the
   * first item of that page (the located item's `index`), so "N of total"
   * stays true.
   */
  loadAt(cursor: string | null, index?: number): Promise<void>;
  /** Refetch the page the buffer started at, in place of what it holds. */
  reload(): Promise<void>;
}

export interface PagerOptions<T> {
  /** Fetch one page; `null` asks for the first. Throwing surfaces on `error`. */
  fetchPage: (cursor: string | null) => Promise<PageResult<T> | null | undefined>;
  /** Stable identity for dedup across pages. */
  keyOf: (item: T) => string;
  /** Runs at the start of loadFirst, before the request. */
  onReset?: () => void;
  /** Runs when loadFirst fails, after `error` is set. */
  onLoadFirstError?: () => void;
  /** Drop fetched items before appending (e.g. already-handled ids). */
  accept?: (item: T) => boolean;
}

export function createPager<T>(opts: PagerOptions<T>): Pager<T> {
  let items = $state<T[]>([]);
  let total = $state<number>(0);
  let loaded = $state<boolean>(false);
  let firstIndex = $state<number>(0);
  let loading = $state<boolean>(false);
  let loadingMore = $state<boolean>(false);
  let error = $state<string | null>(null);
  let nextCursor = $state<string | null>(null);
  // Where the buffer started, so `reload()` can refetch it in place.
  let startCursor: string | null = null;
  const hasMore = $derived(nextCursor !== null);

  // Bumped by every reset (loadFirst / loadAt / reload). loadMore() and the
  // resets each capture the epoch in effect when their fetch started; if a
  // newer reset has since started by the time a fetch resolves, its result is
  // discarded instead of applied. Without this, a loadMore() left in flight
  // when the caller triggers a fresh load (a scrolled bucket, then "Refine")
  // can resolve *after* the reload and silently append its stale page onto
  // the freshly loaded buffer.
  let epoch = 0;

  async function reset(cursor: string | null, index: number): Promise<void> {
    const myEpoch = ++epoch;
    loading = true;
    error = null;
    items = [];
    total = 0;
    loaded = false;
    nextCursor = null;
    opts.onReset?.();
    try {
      const res = await opts.fetchPage(cursor);
      // A newer reset already started (and owns `loading`): leave its result
      // alone rather than overwrite it with our now-stale fetch.
      if (myEpoch !== epoch) return;
      const fresh = uniqueByKey(res?.items ?? [], opts);
      items = fresh;
      total = res?.total ?? fresh.length;
      nextCursor = res?.next_cursor ?? null;
      startCursor = cursor;
      firstIndex = index;
      loaded = true;
    } catch (e) {
      if (myEpoch !== epoch) return;
      error = apiErrorText(e);
      opts.onLoadFirstError?.();
    } finally {
      if (myEpoch === epoch) loading = false;
    }
  }

  return {
    get items() {
      return items;
    },
    set items(next: T[]) {
      items = next;
    },
    get total() {
      return total;
    },
    set total(next: number) {
      total = next;
    },
    get error() {
      return error;
    },
    set error(next: string | null) {
      error = next;
    },
    get loading() {
      return loading;
    },
    get loadingMore() {
      return loadingMore;
    },
    get loaded() {
      return loaded;
    },
    get firstIndex() {
      return firstIndex;
    },
    get hasMore() {
      return hasMore;
    },

    loadFirst: () => reset(null, 0),
    loadAt: (cursor, index = 0) => reset(cursor, index),
    reload: () => reset(startCursor, firstIndex),

    async loadMore(): Promise<void> {
      if (loading || loadingMore || nextCursor === null) return;
      const myEpoch = epoch;
      loadingMore = true;
      try {
        const res = await opts.fetchPage(nextCursor);
        // A reset replaced the buffer while this page was in flight; applying
        // it now would splice a stale page onto the fresh load, so drop it.
        if (myEpoch !== epoch) return;
        // Dedup: the server collection shrinks as crops are relabeled, so a
        // later page can repeat an item an earlier page already returned.
        // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local dedup set consumed synchronously within this call, never stored in reactive state
        const seen = new Set(items.map(opts.keyOf));
        const fresh = (res?.items ?? []).filter(
          (i) => !seen.has(opts.keyOf(i)) && (opts.accept?.(i) ?? true),
        );
        items = [...items, ...fresh];
        total = res?.total ?? total;
        nextCursor = res?.next_cursor ?? null;
      } catch (e) {
        if (myEpoch === epoch) error = apiErrorText(e);
      } finally {
        // Always clear the busy flag, even if superseded — otherwise a
        // discarded stale loadMore() would leave loadingMore stuck `true`
        // and permanently block future loadMore() calls.
        loadingMore = false;
      }
    },
  };
}

/** One served page, filtered by `accept` and reduced to the first item per key. */
function uniqueByKey<T>(
  items: T[],
  opts: { keyOf: (i: T) => string; accept?: (i: T) => boolean },
): T[] {
  const keys: string[] = [];
  return items.filter((i) => {
    if (!(opts.accept?.(i) ?? true)) return false;
    const k = opts.keyOf(i);
    if (keys.includes(k)) return false;
    keys.push(k);
    return true;
  });
}
