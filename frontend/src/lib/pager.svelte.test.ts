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
    const fetchPage = vi.fn(async (page: number) =>
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
        page === 1 ? { items: rows('a', 'b'), total: 4 } : { items: rows('c', 'd'), total: 4 },
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
