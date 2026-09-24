/**
 * Accumulating page-1..N loader shared by the infinite-scroll grids.
 *
 * Every grid on this app repeats the same shape: fetch page 1 into a fresh
 * buffer, then append later pages, deduping by id because the server
 * collection shifts underneath us as crops get relabeled. Hand-rolling that
 * per page is how `loadMore` on /clusters ended up dropping the `max_rank` /
 * `min_blur_ratio` filters that `loadFirst` sent — one `fetchPage(page)`
 * closure per pager makes that class of drift structurally impossible.
 *
 * `items` and `total` stay writable: the pages mutate them optimistically
 * (label, discard, move) between fetches.
 */

export interface PageResult<T> {
  items: T[];
  total?: number | null;
}

export interface Pager<T> {
  items: T[];
  total: number;
  error: string | null;
  readonly loading: boolean;
  readonly loadingMore: boolean;
  readonly loadedPages: number;
  readonly hasMore: boolean;
  /** Reset and fetch page 1. */
  loadFirst(): Promise<void>;
  /** Append the next page; no-op while a fetch is in flight or at the end. */
  loadMore(): Promise<void>;
  /**
   * Reset the buffer and fetch exactly `page`, skipping every page before
   * it. For a deep link that resolves to a page deep into a large queue
   * (DQ-M7: rank 3000 was page 101), `loadMore()` in a 1..page loop means
   * one request per intervening page — 103 requests and 5.7s to land, with
   * the stale page-1 buffer rendered and keyed the whole time. This issues
   * one request. `loadedPages` is set to `page` afterward so a subsequent
   * `loadMore()` continues forward from `page + 1`, not from 2.
   */
  loadPage(page: number): Promise<void>;
}

export interface PagerOptions<T> {
  /** Fetch one page. Throwing surfaces on `error`. */
  fetchPage: (page: number) => Promise<PageResult<T> | null | undefined>;
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
  let loadedPages = $state<number>(0);
  let loading = $state<boolean>(false);
  let loadingMore = $state<boolean>(false);
  let error = $state<string | null>(null);
  const hasMore = $derived(items.length < total);

  // Bumped by every loadFirst(). loadMore() and loadFirst() itself each
  // capture the epoch in effect when their fetch started; if a newer
  // loadFirst() has since started by the time a fetch resolves, its result
  // is discarded instead of applied. Without this, a loadMore() left in
  // flight when the caller triggers a fresh loadFirst() (e.g. /clusters'
  // region bucket view: the user has scrolled a bucket, loading page 2+, then
  // clicks "Refine AHC", whose handler reloads page 1 once the refine POST
  // resolves) can resolve *after* the reload and silently append its stale,
  // pre-reload page onto the freshly loaded buffer — with no error and no
  // visible sign anything went wrong, until a full page reload happens to
  // land cleanly with no competing stale fetch.
  let epoch = 0;

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
    get loadedPages() {
      return loadedPages;
    },
    get hasMore() {
      return hasMore;
    },

    async loadFirst(): Promise<void> {
      const myEpoch = ++epoch;
      loading = true;
      error = null;
      items = [];
      total = 0;
      loadedPages = 0;
      opts.onReset?.();
      try {
        const res = await opts.fetchPage(1);
        // A newer loadFirst() already started (and owns `loading`) — leave
        // its result alone rather than overwrite with our now-stale fetch.
        if (myEpoch !== epoch) return;
        const fresh = (res?.items ?? []).filter((i) => opts.accept?.(i) ?? true);
        items = fresh;
        total = res?.total ?? fresh.length;
        loadedPages = 1;
      } catch (e) {
        if (myEpoch !== epoch) return;
        error = (e as Error).message;
        opts.onLoadFirstError?.();
      } finally {
        if (myEpoch === epoch) loading = false;
      }
    },

    async loadMore(): Promise<void> {
      if (loading || loadingMore || items.length >= total) return;
      const myEpoch = epoch;
      loadingMore = true;
      try {
        const next = loadedPages + 1;
        const res = await opts.fetchPage(next);
        // A loadFirst() reset the buffer while this page was in flight
        // (e.g. a scroll-triggered loadMore() racing a refine-then-reload).
        // Applying it now would silently splice a stale page onto the fresh
        // reload, so drop it instead.
        if (myEpoch !== epoch) return;
        // Dedup: the server collection shrinks as crops are relabeled, so a
        // later page can repeat an item an earlier page already returned.
        const seen = new Set(items.map(opts.keyOf));
        const fresh = (res?.items ?? []).filter(
          (i) => !seen.has(opts.keyOf(i)) && (opts.accept?.(i) ?? true),
        );
        items = [...items, ...fresh];
        total = res?.total ?? total;
        loadedPages = next;
      } catch (e) {
        if (myEpoch === epoch) error = (e as Error).message;
      } finally {
        // Always clear the busy flag, even if superseded — otherwise a
        // discarded stale loadMore() would leave loadingMore stuck `true`
        // and permanently block future loadMore() calls.
        loadingMore = false;
      }
    },

    async loadPage(page: number): Promise<void> {
      const myEpoch = ++epoch;
      loading = true;
      error = null;
      items = [];
      total = 0;
      loadedPages = 0;
      opts.onReset?.();
      try {
        const res = await opts.fetchPage(page);
        if (myEpoch !== epoch) return;
        const fresh = (res?.items ?? []).filter((i) => opts.accept?.(i) ?? true);
        items = fresh;
        total = res?.total ?? fresh.length;
        loadedPages = page;
      } catch (e) {
        if (myEpoch !== epoch) return;
        error = (e as Error).message;
        opts.onLoadFirstError?.();
      } finally {
        if (myEpoch === epoch) loading = false;
      }
    },
  };
}
