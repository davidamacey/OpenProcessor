/**
 * Behavior tests for the /clusters/[id] action controller: moving crops to another cluster (`moveCropIds`).
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

import { moveCropsToCluster, undoCropLabel } from '$lib/api';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';
import { crop, setup } from './clusterControllerTestKit';

afterEach(() => {
  vi.clearAllMocks();
  undoStore.clear();
});

// ---------------------------------------------------------------------
// moveCropIds
// ---------------------------------------------------------------------

describe('moveCropIds', () => {
  it('claims the ids, removes only the moved crop from the grid (leaving another crop in place), and remembers the target on success', async () => {
    vi.mocked(moveCropsToCluster).mockResolvedValue({
      updated: 1,
      updated_ids: ['a'],
      conflicts: [],
    });
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { cropPager, exclusionGuard, rememberTarget, controller } = setup([a, b]);

    const p = controller.moveCropIds(['a'], 99);
    expect(exclusionGuard.accept(a)).toBe(false);
    await p;

    expect(moveCropsToCluster).toHaveBeenCalledWith(['a'], 99);
    expect(cropPager.items.map((c) => c.id)).toEqual(['b']);
    expect(rememberTarget).toHaveBeenCalledWith(99);
    expect(successSpy).toHaveBeenCalledWith('Moved 1 crop → cluster #99.');
  });

  it('pluralizes the success message for a multi-crop move', async () => {
    vi.mocked(moveCropsToCluster).mockResolvedValue({
      updated: 2,
      updated_ids: ['a', 'b'],
      conflicts: [],
    });
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { controller } = setup([a, b]);

    await controller.moveCropIds(['a', 'b'], 99);

    expect(successSpy).toHaveBeenCalledWith('Moved 2 crops → cluster #99.');
  });

  it('releases conflicted ids and reloads instead of toasting plain success', async () => {
    vi.mocked(moveCropsToCluster).mockResolvedValue({
      updated: 1,
      updated_ids: ['a'],
      conflicts: [{ crop_id: 'b', current_source: 'vlm' }],
    });
    const warnSpy = vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { exclusionGuard, loadFirst, controller } = setup([a, b]);

    await controller.moveCropIds(['a', 'b'], 99);

    expect(exclusionGuard.accept(b)).toBe(true);
    expect(exclusionGuard.accept(a)).toBe(false);
    expect(loadFirst).toHaveBeenCalledTimes(1);
    expect(warnSpy).toHaveBeenCalledWith(
      'Moved 1 of 2 crops (1 blocked by worker). Reloading.',
    );
  });

  it('uses the singular wording when exactly one crop was in the (conflicted) request', async () => {
    vi.mocked(moveCropsToCluster).mockResolvedValue({
      updated: 0,
      updated_ids: [],
      conflicts: [{ crop_id: 'a', current_source: 'vlm' }],
    });
    const warnSpy = vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const a = crop('a');
    const { controller } = setup([a]);

    await controller.moveCropIds(['a'], 99);

    expect(warnSpy).toHaveBeenCalledWith(
      'Moved 0 of 1 crop (1 blocked by worker). Reloading.',
    );
  });

  it('treats a missing `conflicts` array as zero conflicts rather than throwing', async () => {
    vi.mocked(moveCropsToCluster).mockResolvedValue({
      updated: 1,
      updated_ids: ['a'],
      conflicts: undefined as unknown as never[],
    });
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const { controller } = setup([a]);

    await expect(controller.moveCropIds(['a'], 99)).resolves.toBeUndefined();

    expect(successSpy).toHaveBeenCalledWith('Moved 1 crop → cluster #99.');
  });

  it('reverts the optimistic removal and releases claims when the move fails outright', async () => {
    vi.mocked(moveCropsToCluster).mockRejectedValue(new Error('down'));
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { cropPager, exclusionGuard, controller } = setup([a, b]);

    await controller.moveCropIds(['a'], 99);

    expect(cropPager.items.map((c) => c.id)).toEqual(['a', 'b']);
    expect(exclusionGuard.accept(a)).toBe(true);
    expect(errorSpy).toHaveBeenCalledWith('Move failed: down');
  });

  it('refuses to move onto the crop’s own cluster', async () => {
    const warnSpy = vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const { controller } = setup([]);

    await controller.moveCropIds(['a'], 42); // getClusterId() -> 42 in setup()

    expect(warnSpy).toHaveBeenCalledWith('Pick a different cluster id.');
    expect(moveCropsToCluster).not.toHaveBeenCalled();
  });

  it('is a no-op with an empty id list', async () => {
    const { controller } = setup([]);

    await controller.moveCropIds([], 99);

    expect(moveCropsToCluster).not.toHaveBeenCalled();
  });

  // M5 (docs/design/interactive-pass-2026-09-24.md): a move is a
  // label-history write like bulkLabel — Z must undo it through the same
  // undoStore/label-undo route. Before this fix moveCropIds never called
  // recordWrites at all, so Z was a silent no-op after a move.
  it('M5: records the server’s updated_ids as one undo entry on success', async () => {
    vi.mocked(moveCropsToCluster).mockResolvedValue({
      updated: 1,
      updated_ids: ['a'],
      conflicts: [],
    });
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const recordWritesSpy = vi.spyOn(undoStore, 'recordWrites');
    const a = crop('a');
    const { controller } = setup([a]);

    await controller.moveCropIds(['a'], 99);

    expect(recordWritesSpy).toHaveBeenCalledWith(['a']);
  });

  it('M5: records only the ids the server actually moved, not the full request, when some conflict', async () => {
    vi.mocked(moveCropsToCluster).mockResolvedValue({
      updated: 1,
      updated_ids: ['a'],
      conflicts: [{ crop_id: 'b', current_source: 'vlm' }],
    });
    vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const recordWritesSpy = vi.spyOn(undoStore, 'recordWrites');
    const a = crop('a');
    const b = crop('b');
    const { controller } = setup([a, b]);

    await controller.moveCropIds(['a', 'b'], 99);

    expect(recordWritesSpy).toHaveBeenCalledWith(['a']);
  });

  it('M5: records nothing (undoStore.recordWrites no-ops) when the move fails outright', async () => {
    vi.mocked(moveCropsToCluster).mockRejectedValue(new Error('down'));
    vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const recordWritesSpy = vi.spyOn(undoStore, 'recordWrites');
    const a = crop('a');
    const { controller } = setup([a]);

    await controller.moveCropIds(['a'], 99);

    expect(recordWritesSpy).not.toHaveBeenCalled();
  });

  it('M5: Z (undoStore.undoLast) actually calls the label-undo route for a moved crop, restoring it via the controller', async () => {
    // A later describe block (`undoLast branch coverage`) permanently
    // stubs `undoStore.undoLast` via `vi.spyOn(...).mockResolvedValue`
    // for its own tests; `vi.clearAllMocks()` in `afterEach` clears call
    // history but not that override. Restore the real implementation so
    // this test exercises the actual undoStore -> undoCropLabel path,
    // not a leftover stub from an unrelated test.
    vi.restoreAllMocks();
    vi.mocked(moveCropsToCluster).mockResolvedValue({
      updated: 1,
      updated_ids: ['a'],
      conflicts: [],
    });
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const restored = crop('a', { cluster_id: 42 });
    vi.mocked(undoCropLabel).mockResolvedValue(restored);
    const a = crop('a');
    const b = crop('b');
    const { cropPager, exclusionGuard, controller } = setup([a, b]);

    await controller.moveCropIds(['a'], 99);
    expect(cropPager.items.map((c) => c.id)).toEqual(['b']);

    await controller.undoLast();

    expect(undoCropLabel).toHaveBeenCalledWith('a');
    expect(exclusionGuard.accept(a)).toBe(true);
    expect(cropPager.items.map((c) => c.id).sort()).toEqual(['a', 'b']);
  });
});
