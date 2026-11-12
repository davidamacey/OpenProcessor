import { describe, expect, it, vi } from 'vitest';
import { createPager, type PageResult } from './pager.svelte';

interface Row {
  id: string;
}

const rows = (...ids: string[]): Row[] => ids.map((id) => ({ id }));
const page = (items: Row[], next: string | null, total = 99): PageResult<Row> => ({
  items,
  total,
  next_cursor: next,
});

describe('createPager', () => {
  it('loadFirst replaces the buffer and records the total and the next cursor', async () => {
    const pager = createPager<Row>({
      fetchPage: async () => page(rows('a', 'b'), 'c1', 5),
      keyOf: (r) => r.id,
    });
    await pager.loadFirst();
    expect(pager.items.map((r) => r.id)).toEqual(['a', 'b']);
    expect(pager.total).toBe(5);
    expect(pager.hasMore).toBe(true);
    expect(pager.loading).toBe(false);
    expect(pager.firstIndex).toBe(0);
  });

  it('appends by next_cursor', async () => {
    const fetchPage = vi.fn(async (cursor: string | null) =>
      cursor === null
        ? page(rows('a', 'b'), 'c1')
        : cursor === 'c1'
          ? page(rows('b', 'c'), 'c2') // 'b' shifted onto page 2
          : page(rows('d'), null),
    );
    const pager = createPager<Row>({ fetchPage, keyOf: (r) => r.id });
    await pager.loadFirst();
    await pager.loadMore();
    await pager.loadMore();
    expect(pager.items.map((r) => r.id)).toEqual(['a', 'b', 'c', 'd']);
    expect(fetchPage.mock.calls.map((c) => c[0])).toEqual([null, 'c1', 'c2']);
    expect(pager.hasMore).toBe(false);
  });

  it('loadMore is a no-op once the server gave no next cursor', async () => {
    const fetchPage = vi.fn(async () => page(rows('a'), null, 1));
    const pager = createPager<Row>({ fetchPage, keyOf: (r) => r.id });
    await pager.loadFirst();
    await pager.loadMore();
    expect(fetchPage).toHaveBeenCalledTimes(1);
    expect(pager.hasMore).toBe(false);
  });

  it('a page that adds nothing still advances by its cursor', async () => {
    const fetchPage = vi.fn(async (cursor: string | null) =>
      cursor === 'c2'
        ? page(rows('z'), null)
        : page(rows('a'), cursor === null ? 'c1' : 'c2'),
    );
    const pager = createPager<Row>({ fetchPage, keyOf: (r) => r.id });
    await pager.loadFirst();
    await pager.loadMore();
    await pager.loadMore();
    expect(pager.items.map((r) => r.id)).toEqual(['a', 'z']);
  });

  it('loadFirst and loadAt drop a key repeated within one served page (keyed grids throw on a duplicate)', async () => {
    const pager = createPager<Row>({
      fetchPage: async () => page(rows('a', 'b', 'a'), null, 3),
      keyOf: (r) => r.id,
    });
    await pager.loadFirst();
    expect(pager.items.map((r) => r.id)).toEqual(['a', 'b']);
    await pager.loadAt('x', 2);
    expect(pager.items.map((r) => r.id)).toEqual(['a', 'b']);
  });

  it('loadAt starts at a located cursor', async () => {
    const seen: (string | null)[] = [];
    const pager = createPager<Row>({
      fetchPage: async (cursor) => {
        seen.push(cursor);
        return page(rows(`${cursor}-a`, `${cursor}-b`), `after-${cursor}`, 200);
      },
      keyOf: (r) => r.id,
    });
    await pager.loadAt('loc', 3000);
    expect(seen).toEqual(['loc']);
    expect(pager.items.map((r) => r.id)).toEqual(['loc-a', 'loc-b']);
    expect(pager.total).toBe(200);
    expect(pager.firstIndex).toBe(3000);
    await pager.loadMore();
    expect(seen).toEqual(['loc', 'after-loc']);
    expect(pager.firstIndex).toBe(3000);
    await pager.loadFirst();
    expect(pager.firstIndex).toBe(0);
  });

  it('loadAt discards the buffer from any prior load', async () => {
    let call = 0;
    const pager = createPager<Row>({
      fetchPage: async () => {
        call++;
        return call === 1 ? page(rows('a', 'b'), 'n') : page(rows('z1', 'z2'), null);
      },
      keyOf: (r) => r.id,
    });
    await pager.loadFirst();
    await pager.loadAt('deep', 20);
    expect(pager.items.map((r) => r.id)).toEqual(['z1', 'z2']);
  });

  it('reload refetches the page the buffer started at', async () => {
    const seen: (string | null)[] = [];
    const pager = createPager<Row>({
      fetchPage: async (cursor) => {
        seen.push(cursor);
        return page(rows(`r${seen.length}`), 'n');
      },
      keyOf: (r) => r.id,
    });
    await pager.loadAt('here', 40);
    await pager.reload();
    expect(seen).toEqual(['here', 'here']);
    expect(pager.firstIndex).toBe(40);
    expect(pager.items.map((r) => r.id)).toEqual(['r2']);
  });

  it('accept() filters fetched items on both first and later pages', async () => {
    const dropped = new Set(['b', 'd']);
    const pager = createPager<Row>({
      fetchPage: async (cursor) =>
        cursor === null ? page(rows('a', 'b'), 'n', 4) : page(rows('c', 'd'), null, 4),
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
      fetchPage: async () => page(rows('a'), null, 1),
      keyOf: (r) => r.id,
      onReset,
    });
    pager.error = 'stale';
    await pager.loadFirst();
    expect(onReset).toHaveBeenCalledTimes(1);
    expect(pager.error).toBeNull();
  });

  it('ignores a stale response after a reset', async () => {
    // The user scrolled a list (loadMore in flight with an old, pre-refine
    // page) and then a reload replaces the buffer. The stale response must not
    // be appended onto the fresh load once it lands.
    let resolveStale: (v: PageResult<Row>) => void = () => {};
    const stale = new Promise<PageResult<Row>>((res) => {
      resolveStale = res;
    });
    let call = 0;
    const pager = createPager<Row>({
      fetchPage: async () => {
        call++;
        if (call === 1) return page(rows('old1', 'old2'), 'n', 4);
        if (call === 2) return stale;
        return page(rows('new1', 'new2'), null, 2);
      },
      keyOf: (r) => r.id,
    });
    await pager.loadFirst();
    const moreDone = pager.loadMore();
    await pager.loadFirst();
    expect(pager.items.map((r) => r.id)).toEqual(['new1', 'new2']);
    resolveStale(page(rows('old2', 'old3'), 'later', 4));
    await moreDone;
    expect(pager.items.map((r) => r.id)).toEqual(['new1', 'new2']);
    expect(pager.total).toBe(2);
    expect(pager.hasMore).toBe(false);
    expect(pager.loadingMore).toBe(false);
  });

  it('a loadAt() in flight beats a slower stale loadFirst() (epoch race)', async () => {
    let resolveFirst: (v: PageResult<Row>) => void = () => {};
    const firstPromise = new Promise<PageResult<Row>>((res) => {
      resolveFirst = res;
    });
    let call = 0;
    const pager = createPager<Row>({
      fetchPage: async () => {
        call++;
        return call === 1 ? firstPromise : page(rows('located'), null, 500);
      },
      keyOf: (r) => r.id,
    });
    const staleLoadFirst = pager.loadFirst();
    await pager.loadAt('deep', 101);
    resolveFirst(page(rows('stale'), null, 3));
    await staleLoadFirst;
    expect(pager.items.map((r) => r.id)).toEqual(['located']);
    expect(pager.firstIndex).toBe(101);
  });

  it('items and total stay writable for optimistic mutations', async () => {
    const pager = createPager<Row>({
      fetchPage: async () => page(rows('a', 'b'), null, 2),
      keyOf: (r) => r.id,
    });
    await pager.loadFirst();
    pager.items = pager.items.filter((r) => r.id !== 'a');
    pager.total = 1;
    expect(pager.items.map((r) => r.id)).toEqual(['b']);
    expect(pager.hasMore).toBe(false);
  });
});
