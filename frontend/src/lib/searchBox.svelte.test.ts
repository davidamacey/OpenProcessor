import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createSemanticSearchBox } from './searchBox.svelte';
import type { SearchCrop } from './types';

function crop(id: string, similarity_score = 0.5): SearchCrop {
  return {
    id,
    source_image_path: '/x.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 1, h: 1 },
    class_id: null,
    class_name: null,
    class_source: null,
    label_source: 'model',
    label_validated: false,
    label_confidence: null,
    cluster_id: null,
    similarity_to_centroid: null,
    cluster_subid: null,
    updated_at: '',
    similarity_score,
  } as unknown as SearchCrop;
}

describe('createSemanticSearchBox', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('debounces typed input before firing a search', async () => {
    const search = vi.fn().mockResolvedValue({ items: [crop('a')], total: 1 });
    const box = createSemanticSearchBox({ search, debounceMs: 300 });

    box.oninput('red widget_a');
    expect(search).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(299);
    expect(search).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1);
    expect(search).toHaveBeenCalledTimes(1);
    expect(search).toHaveBeenCalledWith('red widget_a', expect.any(AbortSignal));
  });

  it('submit() (Enter) fires immediately, bypassing the debounce', async () => {
    const search = vi.fn().mockResolvedValue({ items: [], total: 0 });
    const box = createSemanticSearchBox({ search, debounceMs: 300 });

    box.query = 'blue truck';
    box.submit();
    // No timer advance needed — should already have fired.
    await Promise.resolve();
    await Promise.resolve();
    expect(search).toHaveBeenCalledTimes(1);
    expect(search).toHaveBeenCalledWith('blue truck', expect.any(AbortSignal));
  });

  it('a subsequent keystroke aborts the still-outstanding request for the previous one', async () => {
    let firstSignal: AbortSignal | undefined;
    const search = vi.fn().mockImplementation((_q: string, signal: AbortSignal) => {
      firstSignal ??= signal;
      return new Promise(() => {
        /* never resolves within this test */
      });
    });
    const box = createSemanticSearchBox({ search, debounceMs: 300 });

    box.oninput('red');
    await vi.advanceTimersByTimeAsync(300);
    expect(search).toHaveBeenCalledTimes(1);
    expect(firstSignal?.aborted).toBe(false);

    box.oninput('red widget_a');
    await vi.advanceTimersByTimeAsync(300);
    expect(search).toHaveBeenCalledTimes(2);
    expect(firstSignal?.aborted).toBe(true);
  });

  it('calls onResults with the search response and sets active=true', async () => {
    const items = [crop('a'), crop('b')];
    const search = vi.fn().mockResolvedValue({ items, total: 2 });
    const onResults = vi.fn();
    const box = createSemanticSearchBox({ search, debounceMs: 300, onResults });

    box.submit(); // blank query — no-op
    expect(search).not.toHaveBeenCalled();

    box.query = 'widget_a';
    box.submit();
    await Promise.resolve();
    await Promise.resolve();

    expect(onResults).toHaveBeenCalledWith({ items, total: 2 });
    expect(box.active).toBe(true);
  });

  it('clear() resets to the no-search state and calls onClear when it was active', async () => {
    const search = vi.fn().mockResolvedValue({ items: [crop('a')], total: 1 });
    const onClear = vi.fn();
    const box = createSemanticSearchBox({ search, debounceMs: 300, onClear });

    box.query = 'widget_a';
    box.submit();
    await Promise.resolve();
    await Promise.resolve();
    expect(box.active).toBe(true);

    box.clear();
    expect(box.query).toBe('');
    expect(box.active).toBe(false);
    expect(onClear).toHaveBeenCalledTimes(1);
  });

  it('clear() is a no-op re: onClear when the box was never active', () => {
    const search = vi.fn();
    const onClear = vi.fn();
    const box = createSemanticSearchBox({ search, onClear });
    box.clear();
    expect(onClear).not.toHaveBeenCalled();
  });

  it('emptying the input by hand fires onClear immediately without waiting for the debounce', () => {
    const search = vi.fn().mockResolvedValue({ items: [], total: 0 });
    const onClear = vi.fn();
    const box = createSemanticSearchBox({ search, debounceMs: 300, onClear });

    // Simulate having been active already.
    box.oninput('widget_a');
    box.oninput('');
    expect(search).not.toHaveBeenCalled();
    expect(onClear).not.toHaveBeenCalled(); // wasn't active yet, nothing to clear
  });

  it('empty/error-state: a rejected search surfaces its message on error and never throws', async () => {
    const search = vi.fn().mockRejectedValue(new Error('backend unavailable'));
    const box = createSemanticSearchBox({ search, debounceMs: 300 });

    box.query = 'widget_a';
    box.submit();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(box.error).toBe('backend unavailable');
    expect(box.loading).toBe(false);
  });

  it('empty-state: a zero-hit response still marks active (so the host page shows "no results" not the stale grid)', async () => {
    const search = vi.fn().mockResolvedValue({ items: [], total: 0 });
    const onResults = vi.fn();
    const box = createSemanticSearchBox({ search, debounceMs: 300, onResults });

    box.query = 'zzzznonexistentzzzz';
    box.submit();
    await Promise.resolve();
    await Promise.resolve();

    expect(box.active).toBe(true);
    expect(onResults).toHaveBeenCalledWith({ items: [], total: 0 });
  });

  it('an aborted fetch never sets error or calls onResults for the stale request', async () => {
    const search = vi
      .fn()
      .mockImplementationOnce(
        (_q: string, signal: AbortSignal) =>
          new Promise((_resolve, reject) => {
            signal.addEventListener('abort', () => {
              const err = new DOMException('aborted', 'AbortError');
              reject(err);
            });
          }),
      )
      .mockResolvedValueOnce({ items: [crop('a')], total: 1 });
    const onResults = vi.fn();
    const box = createSemanticSearchBox({ search, debounceMs: 300, onResults });

    box.oninput('red');
    await vi.advanceTimersByTimeAsync(300);
    box.oninput('red widget_a');
    await vi.advanceTimersByTimeAsync(300);
    await Promise.resolve();
    await Promise.resolve();

    expect(box.error).toBeNull();
    expect(onResults).toHaveBeenCalledTimes(1);
    expect(onResults).toHaveBeenCalledWith({ items: [crop('a')], total: 1 });
  });
});
