import { describe, expect, it } from 'vitest';
import { SLOW_LOAD_MS, firstLoadHint, listedText, statusBarText } from './firstLoad';

describe('first load of a cluster', () => {
  it('waits 1.5 s before saying the load is slow', () => {
    expect(SLOW_LOAD_MS).toBe(1500);
  });

  it('names the cluster size when it is known', () => {
    expect(firstLoadHint(4221)).toBe(
      'Ordering 4,221 crops. A large cluster can take a few seconds.',
    );
    expect(firstLoadHint(null)).toBe(
      'Ordering the crops. A large cluster can take a few seconds.',
    );
  });

  it('does not read 0 / 0 listed or "all loaded" while the first page is loading', () => {
    const first = {
      loading: true,
      loadingMore: false,
      hasMore: false,
      itemCount: 0,
      total: 0,
    };
    expect(listedText(first)).toBe('loading…');
    expect(statusBarText(first)).toBe('loading…');
  });

  it('keeps the served counts and the paging words once a page has loaded', () => {
    const base = {
      loading: false,
      loadingMore: false,
      hasMore: true,
      itemCount: 60,
      total: 4221,
    };
    expect(listedText(base)).toBe('60 / 4221 listed');
    expect(statusBarText(base)).toBe('scroll for more');
    expect(statusBarText({ ...base, loadingMore: true })).toBe('loading more…');
    expect(statusBarText({ ...base, hasMore: false })).toBe('all loaded');
  });

  it('an empty cluster that finished loading still reads 0 / 0 listed and all loaded (control)', () => {
    const empty = {
      loading: false,
      loadingMore: false,
      hasMore: false,
      itemCount: 0,
      total: 0,
    };
    expect(listedText(empty)).toBe('0 / 0 listed');
    expect(statusBarText(empty)).toBe('all loaded');
  });
});
