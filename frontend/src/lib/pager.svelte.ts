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
      loading = true;
      error = null;
      items = [];
      total = 0;
      loadedPages = 0;
      opts.onReset?.();
      try {
        const res = await opts.fetchPage(1);
        const fresh = (res?.items ?? []).filter((i) => opts.accept?.(i) ?? true);
        items = fresh;
        total = res?.total ?? fresh.length;
        loadedPages = 1;
      } catch (e) {
        error = (e as Error).message;
        opts.onLoadFirstError?.();
      } finally {
        loading = false;
      }
    },

    async loadMore(): Promise<void> {
      if (loading || loadingMore || items.length >= total) return;
      loadingMore = true;
      try {
        const next = loadedPages + 1;
        const res = await opts.fetchPage(next);
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
        error = (e as Error).message;
      } finally {
        loadingMore = false;
      }
    },
  };
}
