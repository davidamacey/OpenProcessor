/**
 * Behavior tests for the /clusters/[id] action controller: ignore and un-ignore (`ignoreSelected`, `undoIgnore`).
 * Shared fixtures are in `clusterControllerTestKit.ts`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/api_clusters', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_clusters')>('$lib/api_clusters');
  return { ...actual, moveCropsToCluster: vi.fn() };
});
vi.mock('$lib/api_cropWrites', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_cropWrites')>('$lib/api_cropWrites');
  return {
    ...actual,
    putCropLabel: vi.fn(),
    bulkLabel: vi.fn(),
    discardCrop: vi.fn(),
    discardCropsBatch: vi.fn(),
    excludeCrops: vi.fn(),
    unexcludeCrops: vi.fn(),
    vlmDismissCrop: vi.fn(),
    undoCropLabel: vi.fn(),
    undoLabelBatch: vi.fn(),
  };
});

import { excludeCrops, unexcludeCrops } from '$lib/api_cropWrites';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';
import { crop, setup } from './clusterControllerTestKit';

afterEach(() => {
  vi.clearAllMocks();
  undoStore.clear();
});

// ---------------------------------------------------------------------
// ignore / unignore
// ---------------------------------------------------------------------

describe('ignoreSelected / undoIgnore', () => {
  it('excludes only the selected crops from the grid (leaving an unselected crop in place), claims the exclusion guard, and remembers the batch', async () => {
    vi.mocked(excludeCrops).mockResolvedValue({
      excluded: 1,
      updated_ids: [],
      errors: 0,
    });
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b'); // not selected -- must survive the ignore
    const { cropPager, sel, exclusionGuard, controller } = setup([a, b]);
    sel.ids = new Set(['a']);

    await controller.ignoreSelected('blurry');

    expect(excludeCrops).toHaveBeenCalledWith(['a'], 'blurry');
    expect(cropPager.items.map((c) => c.id)).toEqual(['b']);
    expect(exclusionGuard.accept(a)).toBe(false);
    expect(exclusionGuard.accept(b)).toBe(true);
    expect(sel.size).toBe(0);
    expect(successSpy).toHaveBeenCalledWith('Ignored 1 (blurry). Press U to undo.');
  });

  it('defaults the reason to "ignore" and omits the parenthetical tag on the toast', async () => {
    vi.mocked(excludeCrops).mockResolvedValue({
      excluded: 1,
      updated_ids: [],
      errors: 0,
    });
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const { sel, controller } = setup([a]);
    sel.ids = new Set(['a']);

    await controller.ignoreSelected();

    expect(excludeCrops).toHaveBeenCalledWith(['a'], 'ignore');
    expect(successSpy).toHaveBeenCalledWith('Ignored 1. Press U to undo.');
  });

  it('undoIgnore restores the last-ignored batch, releases the exclusion guard, and toasts the exact restored count', async () => {
    vi.mocked(excludeCrops).mockResolvedValue({
      excluded: 1,
      updated_ids: [],
      errors: 0,
    });
    vi.mocked(unexcludeCrops).mockResolvedValue({
      unexcluded: 1,
      updated_ids: [],
      errors: 0,
    });
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const { exclusionGuard, sel, controller } = setup([a]);
    sel.ids = new Set(['a']);

    await controller.ignoreSelected();
    await controller.undoIgnore();

    expect(unexcludeCrops).toHaveBeenCalledWith(['a']);
    expect(exclusionGuard.accept(a)).toBe(true);
    expect(successSpy).toHaveBeenCalledWith('Restored 1. Re-cluster to re-sort them.');

    // A second undoIgnore right after must be a no-op (info toast) --
    // lastExcludedIds must actually have been cleared to [], not left
    // populated (or replaced with a Stryker placeholder id).
    const infoSpy = vi.spyOn(toastStore, 'info').mockImplementation(() => 'x');
    await controller.undoIgnore();
    expect(infoSpy).toHaveBeenCalledWith('Nothing to un-ignore.');
    expect(unexcludeCrops).toHaveBeenCalledTimes(1);
  });

  it('undoIgnore is a no-op (info toast) when there is nothing to un-ignore', async () => {
    const infoSpy = vi.spyOn(toastStore, 'info').mockImplementation(() => 'x');
    const { controller } = setup([]);

    await controller.undoIgnore();

    expect(infoSpy).toHaveBeenCalledWith('Nothing to un-ignore.');
    expect(unexcludeCrops).not.toHaveBeenCalled();
  });

  it('undoIgnore only restores the most recent ignore batch, not an accumulated history', async () => {
    vi.mocked(excludeCrops).mockResolvedValue({
      excluded: 1,
      updated_ids: [],
      errors: 0,
    });
    vi.mocked(unexcludeCrops).mockResolvedValue({
      unexcluded: 1,
      updated_ids: [],
      errors: 0,
    });
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { sel, controller } = setup([a, b]);
    sel.ids = new Set(['a']);
    await controller.ignoreSelected();
    sel.ids = new Set(['b']);
    await controller.ignoreSelected();

    await controller.undoIgnore();

    expect(unexcludeCrops).toHaveBeenCalledWith(['b']);
    expect(unexcludeCrops).toHaveBeenCalledTimes(1);
  });

  it('ignoreSelected is a no-op (info toast) with nothing selected', async () => {
    const infoSpy = vi.spyOn(toastStore, 'info').mockImplementation(() => 'x');
    const { controller } = setup([]);

    await controller.ignoreSelected();

    expect(infoSpy).toHaveBeenCalledWith('Select crops first to ignore.');
    expect(excludeCrops).not.toHaveBeenCalled();
  });

  it('toasts the error message when the exclude request fails, leaving the grid untouched', async () => {
    vi.mocked(excludeCrops).mockRejectedValue(new Error('exclude down'));
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a');
    const { cropPager, sel, exclusionGuard, controller } = setup([a]);
    sel.ids = new Set(['a']);

    await controller.ignoreSelected();

    expect(cropPager.items.map((c) => c.id)).toEqual(['a']);
    expect(exclusionGuard.accept(a)).toBe(true);
    expect(errorSpy).toHaveBeenCalledWith('Ignore failed: exclude down');
  });

  it('toasts the error message when the un-exclude request fails, leaving the exclusion claim in place', async () => {
    vi.mocked(excludeCrops).mockResolvedValue({
      excluded: 1,
      updated_ids: [],
      errors: 0,
    });
    vi.mocked(unexcludeCrops).mockRejectedValue(new Error('unexclude down'));
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a');
    const { exclusionGuard, sel, controller } = setup([a]);
    sel.ids = new Set(['a']);
    await controller.ignoreSelected();

    await controller.undoIgnore();

    expect(exclusionGuard.accept(a)).toBe(false); // still excluded -- release never ran
    expect(errorSpy).toHaveBeenCalledWith('Un-ignore failed: unexclude down');
  });
});
