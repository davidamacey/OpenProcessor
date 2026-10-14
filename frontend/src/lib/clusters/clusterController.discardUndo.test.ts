/**
 * Behavior tests for the /clusters/[id] action controller: discard and Z undo (`discardSelected`, `undoLast`) and `adoptItems`.
 * Shared fixtures are in `clusterControllerTestKit.ts`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return {
    ...actual,
    putCropLabel: vi.fn(),
    bulkLabel: vi.fn(),
    discardCrop: vi.fn(),
    discardCropsBatch: vi.fn(),
    excludeCrops: vi.fn(),
    unexcludeCrops: vi.fn(),
    moveCropsToCluster: vi.fn(),
    vlmDismissCrop: vi.fn(),
    undoCropLabel: vi.fn(),
    undoLabelBatch: vi.fn(),
  };
});

import { discardCrop, discardCropsBatch, undoCropLabel } from '$lib/api';
import { classesStore } from '$stores/classes.svelte';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';
import { crop, setup } from './clusterControllerTestKit';

afterEach(() => {
  vi.clearAllMocks();
  undoStore.clear();
});

// ---------------------------------------------------------------------
// discardSelected -> discard / discard_batch, then Z (undoLast) restores
// ---------------------------------------------------------------------

describe('discardSelected then undoLast', () => {
  it('single selection calls discardCrop; records undo for exactly the discarded id; claims the exclusion guard', async () => {
    vi.mocked(discardCrop).mockResolvedValue(crop('a') as never);
    const recordWritesSpy = vi.spyOn(undoStore, 'recordWrites');
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a');
    const { cropPager, sel, exclusionGuard, controller } = setup([a]);
    sel.ids = new Set(['a']);

    await controller.discardSelected();

    expect(discardCrop).toHaveBeenCalledWith('a');
    expect(cropPager.items).toEqual([]);
    expect(exclusionGuard.accept(a)).toBe(false);
    expect(recordWritesSpy).toHaveBeenCalledWith(['a']);
    expect(successSpy).toHaveBeenCalledWith('Discarded 1. Press Z to undo.');
    // failedCount === 0 here -- must not also fire the failure toast.
    expect(errorSpy).not.toHaveBeenCalled();
  });

  it('multi selection calls discardCropsBatch and keeps failed ids selected for retry', async () => {
    vi.mocked(discardCropsBatch).mockResolvedValue({
      items: [crop('a')],
      discarded: 1,
      conflicts: [],
      not_found: ['b'],
    });
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { cropPager, sel, controller } = setup([a, b]);
    sel.ids = new Set(['a', 'b']);

    await controller.discardSelected();

    expect(discardCropsBatch).toHaveBeenCalledWith(['a', 'b']);
    expect(cropPager.items.map((c) => c.id)).toEqual(['b']);
    expect([...sel.ids]).toEqual(['b']);
    expect(errorSpy).toHaveBeenCalledWith('1 discard(s) failed — still selected.');
  });

  it('undoLast (Z) restores a discarded crop and releases it from the exclusion guard', async () => {
    vi.mocked(discardCrop).mockResolvedValue(crop('a') as never);
    vi.mocked(undoCropLabel).mockResolvedValue(crop('a'));
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { cropPager, sel, exclusionGuard, controller } = setup([a, b]);
    sel.ids = new Set(['a']);

    await controller.discardSelected();
    expect(cropPager.items.map((c) => c.id)).toEqual(['b']);

    await controller.undoLast();

    expect(exclusionGuard.accept(a)).toBe(true);
    expect(cropPager.items.map((c) => c.id)).toEqual(['a', 'b']);
    // discardSelected doesn't touch total (mirrors +page.svelte's
    // original behavior); undoLast's "not present" branch bumps it.
    expect(cropPager.total).toBe(3);
  });

  it('is a no-op when nothing is selected', async () => {
    const { controller } = setup([]);

    await controller.discardSelected();

    expect(discardCrop).not.toHaveBeenCalled();
    expect(discardCropsBatch).not.toHaveBeenCalled();
  });

  it('when the single-crop discard request throws, nothing is removed, nothing is claimed, and only the error toast fires', async () => {
    vi.mocked(discardCrop).mockRejectedValue(new Error('server down'));
    const recordWritesSpy = vi.spyOn(undoStore, 'recordWrites');
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a');
    const { cropPager, sel, exclusionGuard, controller } = setup([a]);
    sel.ids = new Set(['a']);

    await controller.discardSelected();

    // succeededIds must stay [] (not a Stryker-mutated placeholder id) --
    // nothing is filtered out, claimed, or recorded.
    expect(cropPager.items.map((c) => c.id)).toEqual(['a']);
    expect(exclusionGuard.accept(a)).toBe(true);
    expect(recordWritesSpy).toHaveBeenCalledWith([]);
    expect(successSpy).not.toHaveBeenCalled();
    expect(errorSpy).toHaveBeenCalledWith(
      '1 discard(s) failed — still selected: server down',
    );
  });
});

// ---------------------------------------------------------------------
// undoLast: the `if (cropPager.items.some(...))` branch the audit found
// surviving as an `if (false)` mutant (docs/design/test-audit-2026-09-24.md
// §2.2, old +page.svelte:634) -- both branches, in-place replace vs.
// prepend, are exercised explicitly here.
//
// A second, earlier guard in the same function -- `if (crops.length === 0)
// return;` -- still shows up as a surviving `if (false)` mutant under
// Stryker. Same accepted-equivalent reasoning as
// reviewController.test.ts's identical guard: with `crops` empty, the
// `for (const crop of crops)` loop below is a no-op either way, so
// removing the early return cannot produce any observable difference.
// ---------------------------------------------------------------------

describe('undoLast branch coverage', () => {
  it('replaces the crop in place when it is still present in the grid (does not duplicate or reorder)', async () => {
    const restored = crop('b', { class_id: 5 });
    vi.spyOn(undoStore, 'undoLast').mockResolvedValue([restored]);
    const a = crop('a');
    const staleB = crop('b', { class_id: null });
    const c = crop('c');
    const { cropPager, controller } = setup([a, staleB, c]);

    await controller.undoLast();

    expect(cropPager.items.map((x) => x.id)).toEqual(['a', 'b', 'c']);
    expect(cropPager.items[1]).toEqual(restored);
    expect(cropPager.total).toBe(3); // unchanged -- no prepend happened
  });

  it('prepends and increments total when the crop is no longer in the grid', async () => {
    const restored = crop('z');
    vi.spyOn(undoStore, 'undoLast').mockResolvedValue([restored]);
    const a = crop('a');
    const { cropPager, controller } = setup([a]);

    await controller.undoLast();

    expect(cropPager.items.map((x) => x.id)).toEqual(['z', 'a']);
    expect(cropPager.total).toBe(2);
  });

  it('is a no-op when there is nothing to undo', async () => {
    vi.spyOn(undoStore, 'undoLast').mockResolvedValue([]);
    const a = crop('a');
    const { cropPager, controller } = setup([a]);

    await controller.undoLast();

    expect(cropPager.items.map((x) => x.id)).toEqual(['a']);
    expect(cropPager.total).toBe(1);
  });

  // Bug 4: undo also reverses a server-owned label write, so it must
  // refresh classesStore too — but only when something was actually
  // undone (a no-op undo must not fire a needless request).
  it('refreshes classesStore when something was actually undone', async () => {
    const restored = crop('a');
    vi.spyOn(undoStore, 'undoLast').mockResolvedValue([restored]);
    const refreshSpy = vi.spyOn(classesStore, 'refresh').mockResolvedValue();
    const a = crop('a');
    const { controller } = setup([a]);

    await controller.undoLast();

    expect(refreshSpy).toHaveBeenCalledTimes(1);
  });

  it('does not refresh classesStore when there is nothing to undo', async () => {
    vi.spyOn(undoStore, 'undoLast').mockResolvedValue([]);
    const refreshSpy = vi.spyOn(classesStore, 'refresh').mockResolvedValue();
    const a = crop('a');
    const { controller } = setup([a]);

    await controller.undoLast();

    expect(refreshSpy).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------
// adoptItems: the served post-write items of an image Reprocess
// ---------------------------------------------------------------------

describe('adoptItems', () => {
  it('replaces each loaded crop with the served item of the same id and repaints the grid', () => {
    const { controller, cropPager, resetGrid } = setup([
      crop('a', { class_name: 'old' }),
      crop('b'),
      crop('c'),
    ]);
    controller.adoptItems([
      crop('a', { class_name: 'served' }),
      crop('c', { proposed_class_name: 'widget' }),
    ]);
    expect(cropPager.items.map((c) => c.id)).toEqual(['a', 'b', 'c']);
    expect(cropPager.items[0]!.class_name).toBe('served');
    expect(cropPager.items[1]!.class_name).toBeNull();
    expect(cropPager.items[2]!.proposed_class_name).toBe('widget');
    expect(resetGrid).toHaveBeenCalledTimes(1);
  });

  it('ignores an item the grid does not hold (another cluster, or not loaded) and never adds it', () => {
    const { controller, cropPager } = setup([crop('a')]);
    controller.adoptItems([crop('zzz', { class_name: 'elsewhere' })]);
    expect(cropPager.items.map((c) => c.id)).toEqual(['a']);
    expect(cropPager.total).toBe(1);
  });

  it('an empty served list changes nothing, not even the grid', () => {
    const { controller, cropPager, resetGrid } = setup([crop('a')]);
    const before = cropPager.items;
    controller.adoptItems([]);
    expect(cropPager.items).toBe(before);
    expect(resetGrid).not.toHaveBeenCalled();
  });
});
