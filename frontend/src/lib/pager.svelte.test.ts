import { describe, expect, it, vi } from 'vitest';
import { createPager } from './pager.svelte';

interface Row {
  id: string;
}

const rows = (...ids: string[]): Row[] => ids.map((id) => ({ id }));

describe('createPager', () => {
  it('loadFirst replaces the buffer and records the total', async () => {
    const pager = createPager<Row>({
      fetchPage: async () => ({ items: rows('a', 'b'), total: 5 }),
      keyOf: (r) => r.id,
    });
    await pager.loadFirst();
    expect(pager.items.map((r) => r.id)).toEqual(['a', 'b']);
    expect(pager.total).toBe(5);
    expect(pager.hasMore).toBe(true);
    expect(pager.loading).toBe(false);
  });

  it('loadMore appends the next page and dedups by key', async () => {
    const fetchPage = vi.fn(
      async (page: number) =>
        page === 1
          ? { items: rows('a', 'b'), total: 4 }
          : { items: rows('b', 'c'), total: 4 }, // 'b' shifted onto page 2
    );
    const pager = createPager<Row>({ fetchPage, keyOf: (r) => r.id });
    await pager.loadFirst();
    await pager.loadMore();
    expect(pager.items.map((r) => r.id)).toEqual(['a', 'b', 'c']);
    expect(fetchPage).toHaveBeenNthCalledWith(2, 2);
  });

  it('loadFirst and loadPage drop a key repeated within one served page (keyed grids throw on a duplicate)', async () => {
    const pager = createPager<Row>({
      fetchPage: async () => ({ items: rows('a', 'b', 'a'), total: 3 }),
      keyOf: (r) => r.id,
    });
    await pager.loadFirst();
    expect(pager.items.map((r) => r.id)).toEqual(['a', 'b']);
    await pager.loadPage(2);
    expect(pager.items.map((r) => r.id)).toEqual(['a', 'b']);
  });

  it('loadMore is a no-op once everything is loaded', async () => {
    const fetchPage = vi.fn(async () => ({ items: rows('a'), total: 1 }));
    const pager = createPager<Row>({ fetchPage, keyOf: (r) => r.id });
    await pager.loadFirst();
    await pager.loadMore();
    expect(fetchPage).toHaveBeenCalledTimes(1);
    expect(pager.hasMore).toBe(false);
  });

  it('every page goes through the same fetchPage closure', async () => {
    const seen: number[] = [];
    const pager = createPager<Row>({
      fetchPage: async (page) => {
        seen.push(page);
        return { items: rows(`p${page}`), total: 3 };
      },
      keyOf: (r) => r.id,
    });
    await pager.loadFirst();
    await pager.loadMore();
    await pager.loadMore();
    expect(seen).toEqual([1, 2, 3]);
  });

  it('accept() filters fetched items on both first and later pages', async () => {
    const dropped = new Set(['b', 'd']);
    const pager = createPager<Row>({
      fetchPage: async (page) =>
        page === 1
          ? { items: rows('a', 'b'), total: 4 }
          : { items: rows('c', 'd'), total: 4 },
      keyOf: (r) => r.id,
      accept: (r) => !dropped.has(r.id),
    });
    await pager.loadFirst();
    expect(pager.items.map((r) => r.id)).toEqual(['a']);
    await pager.loadMore();
    expect(pager.items.map((r) => r.id)).toEqual(['a', 'c']);
  });

  it('surfaces fetch failures on error and runs the error hook', async () => {
    const onLoadFirstError = vi.fn();
    const pager = createPager<Row>({
      fetchPage: async () => {
        throw new Error('API 500');
      },
      keyOf: (r) => r.id,
      onLoadFirstError,
    });
    await pager.loadFirst();
    expect(pager.error).toBe('API 500');
    expect(pager.items).toEqual([]);
    expect(onLoadFirstError).toHaveBeenCalledTimes(1);
  });

  it('runs onReset before the request and clears the previous error', async () => {
    const onReset = vi.fn();
    const pager = createPager<Row>({
      fetchPage: async () => ({ items: rows('a'), total: 1 }),
      keyOf: (r) => r.id,
      onReset,
    });
    pager.error = 'stale';
    await pager.loadFirst();
    expect(onReset).toHaveBeenCalledTimes(1);
    expect(pager.error).toBeNull();
  });

  it('a loadMore() in flight when loadFirst() reloads does not clobber the fresh page (refine-then-reload race)', async () => {
    // Mirrors /clusters' region bucket view: the user has scrolled a bucket
    // (loadMore() in flight fetching an old, pre-refine page) and then
    // triggers "Refine AHC", whose handler calls loadFirst() to reload page 1
    // with the freshly-refined data. If the stale loadMore() resolves AFTER
    // loadFirst()'s own fetch, it must not be allowed to append its
    // pre-refine items onto (or otherwise corrupt) the just-reloaded state.
    let resolveLoadMorePage: (v: { items: Row[]; total: number }) => void;
    const loadMorePagePromise = new Promise<{ items: Row[]; total: number }>((res) => {
      resolveLoadMorePage = res;
    });
    let call = 0;
    const pager = createPager<Row>({
      fetchPage: async () => {
        call++;
        if (call === 1) {
          // Initial loadFirst(): page 1 of the pre-refine bucket.
          return { items: rows('old1', 'old2'), total: 4 };
        }
        if (call === 2) {
          // loadMore() for page 2 -- held open until after the reload below.
          return loadMorePagePromise;
        }
        // The refine handler's own loadFirst(): fresh, post-refine page 1.
        return { items: rows('new1', 'new2'), total: 2 };
      },
      keyOf: (r) => r.id,
    });

    await pager.loadFirst();
    const loadMoreDone = pager.loadMore(); // page 2 fetch now in flight, unresolved

    // "Refine AHC" reloads page 1 with fresh data while loadMore() is still
    // pending -- this must fully win, exactly like production's
    // runRefineCluster() -> loadFirst() after the refine POST.
    await pager.loadFirst();
    expect(pager.items.map((r) => r.id)).toEqual(['new1', 'new2']);
    expect(pager.total).toBe(2);

    // Now let the stale loadMore() response land late.
    resolveLoadMorePage!({ items: rows('old2', 'old3'), total: 4 });
    await loadMoreDone;

    // The stale page must not have been appended onto the fresh reload.
    expect(pager.items.map((r) => r.id)).toEqual(['new1', 'new2']);
    expect(pager.total).toBe(2);
    expect(pager.loadedPages).toBe(1);
    expect(pager.loadingMore).toBe(false);
  });

  it('loadPage fetches only the requested page, skipping every page before it (DQ-M7)', async () => {
    const seen: number[] = [];
    const pager = createPager<Row>({
      fetchPage: async (page) => {
        seen.push(page);
        return { items: rows(`p${page}a`, `p${page}b`), total: 200 };
      },
      keyOf: (r) => r.id,
    });
    await pager.loadPage(101);
    expect(seen).toEqual([101]);
    expect(pager.items.map((r) => r.id)).toEqual(['p101a', 'p101b']);
    expect(pager.total).toBe(200);
    expect(pager.loadedPages).toBe(101);
  });

  it('F8 D6: firstPage records which served page the buffer starts at', async () => {
    const pager = createPager<Row>({
      fetchPage: async (page: number) => ({ items: rows(`p${page}`), total: 100 }),
      keyOf: (r) => r.id,
    });
    await pager.loadPage(3);
    expect(pager.firstPage).toBe(3);
    await pager.loadMore();
    expect(pager.firstPage).toBe(3);
    await pager.loadFirst();
    expect(pager.firstPage).toBe(1);
  });

  it('loadPage discards the buffer from any prior loadFirst/loadMore', async () => {
    let call = 0;
    const pager = createPager<Row>({
      fetchPage: async () => {
        call++;
        if (call === 1) return { items: rows('a', 'b'), total: 50 };
        return { items: rows('z1', 'z2'), total: 50 };
      },
      keyOf: (r) => r.id,
    });
    await pager.loadFirst();
    expect(pager.items.map((r) => r.id)).toEqual(['a', 'b']);
    await pager.loadPage(20);
    expect(pager.items.map((r) => r.id)).toEqual(['z1', 'z2']);
    expect(pager.loadedPages).toBe(20);
  });

  it('loadPage lets a subsequent loadMore() continue forward from page + 1, not page 2', async () => {
    const seen: number[] = [];
    const pager = createPager<Row>({
      fetchPage: async (page) => {
        seen.push(page);
        return { items: rows(`p${page}`), total: 500 };
      },
      keyOf: (r) => r.id,
    });
    await pager.loadPage(101);
    await pager.loadMore();
    expect(seen).toEqual([101, 102]);
  });

  it('a loadPage() in flight beats a slower stale loadFirst() (epoch race)', async () => {
    let resolveFirst: (v: { items: Row[]; total: number }) => void;
    const firstPromise = new Promise<{ items: Row[]; total: number }>((res) => {
      resolveFirst = res;
    });
    let call = 0;
    const pager = createPager<Row>({
      fetchPage: async () => {
        call++;
        if (call === 1) return firstPromise; // stale loadFirst(), held open
        return { items: rows('located'), total: 500 }; // loadPage(), resolves first
      },
      keyOf: (r) => r.id,
    });
    const staleLoadFirst = pager.loadFirst();
    await pager.loadPage(101);
    expect(pager.items.map((r) => r.id)).toEqual(['located']);
    resolveFirst!({ items: rows('stale'), total: 3 });
    await staleLoadFirst;
    // The stale loadFirst() resolved after loadPage() won the epoch race —
    // its result must not clobber the located page.
    expect(pager.items.map((r) => r.id)).toEqual(['located']);
    expect(pager.loadedPages).toBe(101);
  });

  it('items and total stay writable for optimistic mutations', async () => {
    const pager = createPager<Row>({
      fetchPage: async () => ({ items: rows('a', 'b'), total: 2 }),
      keyOf: (r) => r.id,
    });
    await pager.loadFirst();
    pager.items = pager.items.filter((r) => r.id !== 'a');
    pager.total = 1;
    expect(pager.items.map((r) => r.id)).toEqual(['b']);
    expect(pager.hasMore).toBe(false);
  });
});
