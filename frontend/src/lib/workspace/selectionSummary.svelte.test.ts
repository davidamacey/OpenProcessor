import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';

vi.mock('$lib/api_select', () => ({ getSelectionSummary: vi.fn() }));
import { getSelectionSummary } from '$lib/api_select';
import { createSelectionSummary } from './selectionSummary.svelte';
import type { ItemSelection, SelectionSummary } from '$lib/types_itemFilter';

const SERVED: SelectionSummary = {
  detections: 3,
  images: 2,
  clusters: 1,
  locked: 1,
  in_holdout: 0,
  by_class: [],
  located: 0,
  join_truncated: false,
  summary_text: '3 detections in 2 images · 1 locked, skipped',
};

let cleanup: (() => void) | undefined;
afterEach(() => {
  cleanup?.();
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe('createSelectionSummary', () => {
  it('asks the server for the selection and keeps its text verbatim', async () => {
    vi.useFakeTimers();
    vi.mocked(getSelectionSummary).mockResolvedValue(SERVED);
    let sel = $state<ItemSelection | null>({ crop_ids: ['a', 'b', 'c'] });
    let state!: ReturnType<typeof createSelectionSummary>;
    cleanup = $effect.root(() => {
      state = createSelectionSummary(() => sel);
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(200);
    expect(state.summary?.summary_text).toBe(SERVED.summary_text);
    expect(vi.mocked(getSelectionSummary).mock.calls[0]![0]).toEqual({
      crop_ids: ['a', 'b', 'c'],
    });
    sel = null;
    flushSync();
    expect(state.summary).toBeNull();
  });
});
