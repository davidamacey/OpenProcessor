/**
 * Words for the first page of a cluster. A large cluster's first page can take
 * seconds (the backend orders every member), so the view says what it is
 * doing instead of reading "0 / 0 listed" and "all loaded".
 */
export const SLOW_LOAD_MS = 1500;

export interface PagerState {
  loading: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  itemCount: number;
  total: number;
}

export function firstLoadHint(size: number | null): string {
  const what = size == null ? 'the crops' : `${size.toLocaleString('en-US')} crops`;
  return `Ordering ${what}. A large cluster can take a few seconds.`;
}

const isFirstLoad = (s: PagerState): boolean => s.loading && s.itemCount === 0;

export function listedText(s: PagerState): string {
  return isFirstLoad(s) ? 'loading…' : `${s.itemCount} / ${s.total} listed`;
}

export function statusBarText(s: PagerState): string {
  if (isFirstLoad(s)) return 'loading…';
  if (s.loadingMore) return 'loading more…';
  return s.hasMore ? 'scroll for more' : 'all loaded';
}
