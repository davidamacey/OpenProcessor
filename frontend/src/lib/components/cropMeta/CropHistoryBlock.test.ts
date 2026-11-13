import { flushSync } from 'svelte';
import { describe, expect, it } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import type { CropHistoryEntry } from '$lib/types';
import CropHistoryBlock from './CropHistoryBlock.svelte';

const entry = (i: number): CropHistoryEntry =>
  ({
    writer: `writer_${i}`,
    at: `2026-09-24T16:26:0${i}+00:00`,
    class_name: `class_${i}`,
    class_source: null,
  }) as unknown as CropHistoryEntry;

const rows = (el: HTMLElement) =>
  [...el.querySelectorAll('[data-testid="history-entry"]')].map(
    (r) => r.textContent ?? '',
  );

describe('CropHistoryBlock', () => {
  it('shows the last three writes (the served order is oldest first) and a Show all', () => {
    const el = mountUi(CropHistoryBlock, {
      historyEntries: [0, 1, 2, 3, 4].map(entry),
      historyError: null,
      historyLoading: false,
    });
    expect(rows(el)).toHaveLength(3);
    expect(rows(el)[0]).toContain('writer_2');
    expect(rows(el)[2]).toContain('writer_4');
    const all = el.querySelector<HTMLButtonElement>('[data-testid="history-show-all"]')!;
    expect(all.textContent).toContain('Show all 5');
    all.click();
    flushSync();
    expect(rows(el)).toHaveLength(5);
    expect(el.querySelector('[data-testid="history-show-all"]')).toBeNull();
  });

  it('offers no Show all when there are three writes or fewer', () => {
    const el = mountUi(CropHistoryBlock, {
      historyEntries: [0, 1, 2].map(entry),
      historyError: null,
      historyLoading: false,
    });
    expect(rows(el)).toHaveLength(3);
    expect(el.querySelector('[data-testid="history-show-all"]')).toBeNull();
  });
});
