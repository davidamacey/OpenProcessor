/**
 * Behavior tests for the /clusters/[id] action controller: moving crops to another cluster (`moveCropIds`).
 * Shared fixtures are in `clusterControllerTestKit.ts`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/api_clusters', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_clusters')>('$lib/api_clusters');
  return { ...actual, moveCropsToCluster: vi.fn() };
});
vi.mock('$lib/api_jobs', () => ({ getJob: vi.fn() }));
vi.mock('$lib/api_cropRead', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_cropRead')>('$lib/api_cropRead');
  return { ...actual, getCropsByIds: vi.fn() };
});
vi.mock('$lib/labelOps/runLabelOp', async () => {
  const actual = await vi.importActual<typeof import('$lib/labelOps/runLabelOp')>(
    '$lib/labelOps/runLabelOp',
  );
  return { ...actual, undoLabelOpToEnd: vi.fn() };
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

import { moveCropsToCluster } from '$lib/api_clusters';
import { getCropsByIds } from '$lib/api_cropRead';
import { undoLabelOpToEnd } from '$lib/labelOps/runLabelOp';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';
import { crop, mockLabelOp, setup } from './clusterControllerTestKit';

afterEach(() => {
  vi.clearAllMocks();
  undoStore.clear();
});

// ---------------------------------------------------------------------
// moveCropIds
// ---------------------------------------------------------------------

describe('moveCropIds', () => {
  it('claims the ids, removes only the moved crop from the grid (leaving another crop in place), and remembers the target on success', async () => {
    mockLabelOp(moveCropsToCluster, { updated: 1, conflicts: 0 });
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { cropPager, exclusionGuard, rememberTarget, controller } = setup([a, b]);

    const p = controller.moveCropIds(['a'], 'uid-99');
    expect(exclusionGuard.accept(a)).toBe(false);
    await p;

    expect(moveCropsToCluster).toHaveBeenCalledWith(['a'], 'uid-99');
    expect(cropPager.items.map((c) => c.id)).toEqual(['b']);
    expect(rememberTarget).toHaveBeenCalledWith('uid-99');
    expect(successSpy).toHaveBeenCalledWith('Moved 1 crop → cluster uid-99.');
  });

  it('pluralizes the success message for a multi-crop move', async () => {
    mockLabelOp(moveCropsToCluster, { updated: 2, conflicts: 0 });
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { controller } = setup([a, b]);

    await controller.moveCropIds(['a', 'b'], 'uid-99');

    expect(successSpy).toHaveBeenCalledWith('Moved 2 crops → cluster uid-99.');
  });

  it('releases conflicted ids and reloads instead of toasting plain success', async () => {
    mockLabelOp(moveCropsToCluster, { updated: 1, conflicts: 1 });
    const warnSpy = vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { exclusionGuard, loadFirst, controller } = setup([a, b]);

    await controller.moveCropIds(['a', 'b'], 'uid-99');

    // Everything is let back in; the reload shows what the server holds.
    expect(exclusionGuard.accept(a)).toBe(true);
    expect(exclusionGuard.accept(b)).toBe(true);
    expect(loadFirst).toHaveBeenCalledTimes(1);
    expect(warnSpy).toHaveBeenCalledWith(
      'Moved 1 of 2 crops (1 blocked by worker). Reloading.',
    );
  });

  it('uses the singular wording when exactly one crop was in the (conflicted) request', async () => {
    mockLabelOp(moveCropsToCluster, { updated: 0, conflicts: 1 });
    const warnSpy = vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const a = crop('a');
    const { controller } = setup([a]);

    await controller.moveCropIds(['a'], 'uid-99');

    expect(warnSpy).toHaveBeenCalledWith(
      'Moved 0 of 1 crop (1 blocked by worker). Reloading.',
    );
  });

  it('follows the queued job to its end before it reports the move', async () => {
    mockLabelOp(moveCropsToCluster, { updated: 1, conflicts: 0 });
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const { controller } = setup([a]);

    await controller.moveCropIds(['a'], 'uid-99');

    expect(successSpy).toHaveBeenCalledWith('Moved 1 crop → cluster uid-99.');
  });

  it('reverts the optimistic removal and releases claims when the move fails outright', async () => {
    vi.mocked(moveCropsToCluster).mockRejectedValue(new Error('down'));
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { cropPager, exclusionGuard, controller } = setup([a, b]);

    await controller.moveCropIds(['a'], 'uid-99');

    expect(cropPager.items.map((c) => c.id)).toEqual(['a', 'b']);
    expect(exclusionGuard.accept(a)).toBe(true);
    expect(errorSpy).toHaveBeenCalledWith('Move failed: down');
  });

  it('refuses to move onto the crop’s own cluster', async () => {
    const warnSpy = vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const { controller } = setup([]);

    await controller.moveCropIds(['a'], 'cluster-uid-42'); // getClusterUid() in setup()

    expect(warnSpy).toHaveBeenCalledWith('Pick a different cluster.');
    expect(moveCropsToCluster).not.toHaveBeenCalled();
  });

  it('is a no-op with an empty id list', async () => {
    const { controller } = setup([]);

    await controller.moveCropIds([], 'uid-99');

    expect(moveCropsToCluster).not.toHaveBeenCalled();
  });

  // M5 (interactive pass): a move is a
  // label-history write like bulkLabel — Z must undo it through the same
  // undoStore/label-undo route. Before this fix moveCropIds never called
  // recordWrites at all, so Z was a silent no-op after a move.
  it('M5: records the recorded operation as one undo entry on success', async () => {
    mockLabelOp(moveCropsToCluster, { updated: 1, conflicts: 0 });
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const recordSpy = vi.spyOn(undoStore, 'recordLabelOp');
    const a = crop('a');
    const { controller } = setup([a]);

    await controller.moveCropIds(['a'], 'uid-99');

    expect(recordSpy).toHaveBeenCalledWith('op1', ['a']);
  });

  it('M5: records the operation (the server undoes exactly what it moved) when some conflict', async () => {
    mockLabelOp(moveCropsToCluster, { updated: 1, conflicts: 1 });
    vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const recordSpy = vi.spyOn(undoStore, 'recordLabelOp');
    const a = crop('a');
    const b = crop('b');
    const { controller } = setup([a, b]);

    await controller.moveCropIds(['a', 'b'], 'uid-99');

    expect(recordSpy).toHaveBeenCalledWith('op1', ['a', 'b']);
  });

  it('M5: records nothing (nothing is recorded) when the move fails outright', async () => {
    vi.mocked(moveCropsToCluster).mockRejectedValue(new Error('down'));
    vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const recordSpy = vi.spyOn(undoStore, 'recordLabelOp');
    const a = crop('a');
    const { controller } = setup([a]);

    await controller.moveCropIds(['a'], 'uid-99');

    expect(recordSpy).not.toHaveBeenCalled();
  });

  it('M5: Z (undoStore.undoLast) undoes the move by operation id and restores the crop in the grid', async () => {
    // A later describe block (`undoLast branch coverage`) permanently
    // stubs `undoStore.undoLast`; restore the real implementation so this
    // test exercises the actual undoStore -> label-op undo path.
    vi.restoreAllMocks();
    mockLabelOp(moveCropsToCluster, { updated: 1, conflicts: 0 });
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const restored = crop('a');
    vi.mocked(undoLabelOpToEnd).mockResolvedValue({
      restored: 1,
      kept_changed: 0,
    } as never);
    vi.mocked(getCropsByIds).mockResolvedValue([restored]);
    const a = crop('a');
    const b = crop('b');
    const { cropPager, exclusionGuard, controller } = setup([a, b]);

    await controller.moveCropIds(['a'], 'uid-99');
    expect(cropPager.items.map((c) => c.id)).toEqual(['b']);

    await controller.undoLast();

    expect(undoLabelOpToEnd).toHaveBeenCalledWith('op1', false);
    expect(getCropsByIds).toHaveBeenCalledWith(['a']);
    expect(exclusionGuard.accept(a)).toBe(true);
    expect(cropPager.items.map((c) => c.id).sort()).toEqual(['a', 'b']);
  });
});
