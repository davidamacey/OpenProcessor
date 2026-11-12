/**
 * Behavior tests for the /clusters/[id] action controller: the VLM suggestion actions (`acceptVlmForCrop`, `rejectVlmForCrop`, `acceptAllVlmOnPage`).
 * Shared fixtures are in `clusterControllerTestKit.ts`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/api_clusters', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_clusters')>('$lib/api_clusters');
  return { ...actual, moveCropsToCluster: vi.fn() };
});
vi.mock('$lib/api_jobs', () => ({ getJob: vi.fn() }));
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

import { bulkLabel, putCropLabel, vlmDismissCrop } from '$lib/api_cropWrites';
import { ApiError } from '$lib/api_errors';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';
import { getJob } from '$lib/api_jobs';
import { ACCEPTED, crop, doneJob, setup } from './clusterControllerTestKit';

afterEach(() => {
  vi.clearAllMocks();
  undoStore.clear();
});

// ---------------------------------------------------------------------
// acceptVlmForCrop / rejectVlmForCrop
// ---------------------------------------------------------------------

describe('acceptVlmForCrop', () => {
  // A label does not move an item between clusters, so accepting a
  // suggestion applies it in place whatever class it names.
  it('applies the suggestion in place, without removing the crop, and records undo', async () => {
    vi.mocked(putCropLabel).mockResolvedValue({
      crop_id: 'a',
      class_id: 9,
      class_name: 'van',
      op_id: 'op-a',
    });
    const recordLabelOpSpy = vi.spyOn(undoStore, 'recordLabelOp');
    const a = crop('a', { vlm_suggested_class_id: 9, vlm_suggested_class_name: 'van' });
    const { cropPager, exclusionGuard, controller } = setup([a]);

    await controller.acceptVlmForCrop(a);

    expect(cropPager.items).toHaveLength(1);
    expect(cropPager.items[0]!.class_id).toBe(9);
    // ?? null (not && null): a truthy name must survive, not collapse to null.
    expect(cropPager.items[0]!.class_name).toBe('van');
    expect(exclusionGuard.accept(a)).toBe(true);
    expect(putCropLabel).toHaveBeenCalledWith('a', 9);
    expect(recordLabelOpSpy).toHaveBeenCalledWith('op-a', ['a']);
  });

  it('reverts the optimistic label (class restored) when the write fails', async () => {
    vi.mocked(putCropLabel).mockRejectedValue(new Error('nope'));
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a', { vlm_suggested_class_id: 9, vlm_suggested_class_name: 'van' });
    const { cropPager, controller } = setup([a]);

    await controller.acceptVlmForCrop(a);

    expect(cropPager.items).toEqual([a]);
    expect(cropPager.total).toBe(1);
    expect(errorSpy).toHaveBeenCalledWith('Accept VLM suggestion failed: nope');
  });

  it('reverts the optimistic label in place when the write fails, for a same-cluster accept', async () => {
    vi.mocked(putCropLabel).mockRejectedValue(new Error('nope'));
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a', {
      vlm_suggested_class_id: 42,
      vlm_suggested_class_name: 'widget_b',
    });
    const { cropPager, controller } = setup([a]);

    await controller.acceptVlmForCrop(a);

    expect(cropPager.items[0]!.class_id).toBeNull();
    expect(errorSpy).toHaveBeenCalledWith('Accept VLM suggestion failed: nope');
  });

  it('is a no-op when there is no suggestion', async () => {
    const a = crop('a');
    const { controller } = setup([a]);

    await controller.acceptVlmForCrop(a);

    expect(putCropLabel).not.toHaveBeenCalled();
  });
});

describe('rejectVlmForCrop', () => {
  it('renders the item the server returns', async () => {
    const updated = crop('a', { vlm_suggested_class_id: null });
    vi.mocked(vlmDismissCrop).mockResolvedValue(updated);
    const a = crop('a', { vlm_suggested_class_id: 9 });
    const { cropPager, controller } = setup([a]);

    await controller.rejectVlmForCrop(a);

    expect(cropPager.items[0]).toEqual(updated);
  });

  it('replaces only the target crop, leaving every other crop in the grid untouched', async () => {
    const updated = crop('a', { vlm_suggested_class_id: null });
    vi.mocked(vlmDismissCrop).mockResolvedValue(updated);
    const a = crop('a', { vlm_suggested_class_id: 9 });
    const b = crop('b', { vlm_suggested_class_id: 5 });
    const { cropPager, controller } = setup([a, b]);

    await controller.rejectVlmForCrop(a);

    expect(cropPager.items.map((c) => c.id)).toEqual(['a', 'b']);
    expect(cropPager.items[1]!.vlm_suggested_class_id).toBe(5); // untouched
  });

  it('shows an info toast (not an error) on 409', async () => {
    vi.mocked(vlmDismissCrop).mockRejectedValue(new ApiError(409, '/x', {}));
    const infoSpy = vi.spyOn(toastStore, 'info').mockImplementation(() => 'x');
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a');
    const { controller } = setup([a]);

    await controller.rejectVlmForCrop(a);

    expect(infoSpy).toHaveBeenCalledWith('No VLM suggestion to reject.');
    expect(errorSpy).not.toHaveBeenCalled();
  });

  it('requires BOTH an ApiError instance AND status 409 -- an ApiError with a different status is still an error toast', async () => {
    vi.mocked(vlmDismissCrop).mockRejectedValue(new ApiError(500, '/x', {}));
    const infoSpy = vi.spyOn(toastStore, 'info').mockImplementation(() => 'x');
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a');
    const { controller } = setup([a]);

    await controller.rejectVlmForCrop(a);

    expect(infoSpy).not.toHaveBeenCalled();
    expect(errorSpy).toHaveBeenCalled();
  });

  it('shows an error toast on a non-409 failure', async () => {
    vi.mocked(vlmDismissCrop).mockRejectedValue(new Error('down'));
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a');
    const { controller } = setup([a]);

    await controller.rejectVlmForCrop(a);

    expect(errorSpy).toHaveBeenCalledWith('Reject VLM suggestion failed: down');
  });

  // M6/V1 (interactive pass): Z must reverse a
  // Reject-VLM the same way it reverses a label write.
  it('records a vlm_dismiss undo entry on success, so Z can reverse it', async () => {
    const updated = crop('a', { vlm_suggested_class_id: null });
    vi.mocked(vlmDismissCrop).mockResolvedValue(updated);
    const a = crop('a', { vlm_suggested_class_id: 9 });
    const { controller } = setup([a]);
    const recordSpy = vi.spyOn(undoStore, 'recordVlmDismiss');

    await controller.rejectVlmForCrop(a);

    expect(recordSpy).toHaveBeenCalledWith('a');
  });

  it('records no undo entry on a 409 (nothing was actually dismissed)', async () => {
    vi.mocked(vlmDismissCrop).mockRejectedValue(new ApiError(409, '/x', {}));
    vi.spyOn(toastStore, 'info').mockImplementation(() => 'x');
    const a = crop('a');
    const { controller } = setup([a]);
    const recordSpy = vi.spyOn(undoStore, 'recordVlmDismiss');

    await controller.rejectVlmForCrop(a);

    expect(recordSpy).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------
// acceptAllVlmOnPage: groups by class, rolls back only failed groups
// ---------------------------------------------------------------------

// Stryker flags `if (prior) revertLocalLabel(prior);` in
// `acceptAllVlmOnPage` (clusterController.svelte.ts:311, ConditionalExpression
// mutant forcing `if (true)`) as surviving. Accepted equivalent: `priors`
// is populated with an entry for every id in every group before any
// request fires, and `failedIds` is only ever populated from those same
// groups' ids -- `priors.get(id)` can never actually be undefined for an
// id reachable here, so the guard is structurally always-true already.
describe('acceptAllVlmOnPage', () => {
  it('groups targets by suggested class and issues one bulkLabel per group', async () => {
    vi.mocked(bulkLabel).mockImplementation(async () => ACCEPTED);
    vi.mocked(getJob).mockImplementation(async (id) => doneJob({ updated: 1 }, id));
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const recordSpy = vi.spyOn(undoStore, 'recordLabelOp');
    const a = crop('a', { vlm_suggested_class_id: 1, vlm_suggested_class_name: 'x1' });
    const b = crop('b', { vlm_suggested_class_id: 1, vlm_suggested_class_name: 'x1' });
    const c = crop('c', { vlm_suggested_class_id: 2, vlm_suggested_class_name: 'x2' });
    const { cropPager, controller, setVisible } = setup([a, b, c]);
    setVisible([a, b, c]);

    await controller.acceptAllVlmOnPage();

    expect(bulkLabel).toHaveBeenCalledTimes(2);
    expect(bulkLabel).toHaveBeenCalledWith(['a', 'b'], 1);
    expect(bulkLabel).toHaveBeenCalledWith(['c'], 2);
    // One recorded operation per group, so Z reverses each by id.
    expect(recordSpy).toHaveBeenCalledWith('op1', ['a', 'b']);
    expect(recordSpy).toHaveBeenCalledWith('op1', ['c']);
    expect(successSpy).toHaveBeenCalledWith('Accepted 3 suggestions.');
    // A label does not move a crop out of its cluster: all three stay,
    // each showing its accepted class.
    expect(cropPager.items.map((x) => [x.id, x.class_id])).toEqual([
      ['a', 1],
      ['b', 1],
      ['c', 2],
    ]);
  });

  // The final error toast's `lastError ? `: ${lastError}` : '.'` else
  // branch (clusterController.svelte.ts:314) is unreachable, not
  // uncovered: `lastError` is only ever read after `failedIds.length ===
  // 0` has already returned early, and the only place that pushes onto
  // `failedIds` also unconditionally sets `lastError` in the same catch
  // block right before it -- so by the time the toast below runs,
  // `lastError` is always non-null. A test forcing the `'.'` branch would
  // have to fake `failedIds` non-empty with `lastError` still null, which
  // isn't reachable through the public `acceptAllVlmOnPage()` surface.
  it('rolls back only the crops in a failed group, keeping the succeeded group labeled', async () => {
    vi.mocked(bulkLabel).mockImplementation(async (_ids, classId) => {
      if (classId === 2) throw new Error('group 2 failed');
      return ACCEPTED;
    });
    vi.mocked(getJob).mockImplementation(async (id) => doneJob({ updated: 1 }, id));
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a', { vlm_suggested_class_id: 1, vlm_suggested_class_name: 'x1' });
    const c = crop('c', { vlm_suggested_class_id: 2, vlm_suggested_class_name: 'x2' });
    const { cropPager, controller, setVisible } = setup([a, c]);
    setVisible([a, c]);

    await controller.acceptAllVlmOnPage();

    const byId = new Map(cropPager.items.map((x) => [x.id, x]));
    expect(byId.get('a')?.class_id).toBe(1);
    expect(byId.get('c')).toEqual(c);
    expect(errorSpy).toHaveBeenCalledWith(
      'Accepted 1, failed 1 (reverted): group 2 failed',
    );
  });

  it('is a no-op with an info toast when nothing on the page has a suggestion', async () => {
    const infoSpy = vi.spyOn(toastStore, 'info').mockImplementation(() => 'x');
    const a = crop('a');
    const { controller, setVisible } = setup([a]);
    setVisible([a]);

    await controller.acceptAllVlmOnPage();

    expect(infoSpy).toHaveBeenCalledWith('No VLM suggestions on this page.');
    expect(bulkLabel).not.toHaveBeenCalled();
  });

  it('excludes already-class_validated crops even when they carry a suggestion', async () => {
    const a = crop('a', {
      vlm_suggested_class_id: 1,
      vlm_suggested_class_name: 'x1',
      class_validated: true,
    });
    const infoSpy = vi.spyOn(toastStore, 'info').mockImplementation(() => 'x');
    const { controller, setVisible } = setup([a]);
    setVisible([a]);

    await controller.acceptAllVlmOnPage();

    expect(infoSpy).toHaveBeenCalledWith('No VLM suggestions on this page.');
    expect(bulkLabel).not.toHaveBeenCalled();
  });
});
