/**
 * Behavior tests for the /clusters/[id] action controller: class assignment (`assignClassToSelected`) and the drop-on-class / class-hotkey path (`handleClassDrop`).
 * Shared fixtures are in `clusterControllerTestKit.ts`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { RegistryClass } from '$lib/types_classes';

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

import { bulkLabel, putCropLabel } from '$lib/api_cropWrites';
import { classesStore } from '$stores/classes.svelte';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';
import { crop, mockLabelOp, setup } from './clusterControllerTestKit';

afterEach(() => {
  vi.clearAllMocks();
  undoStore.clear();
});

// ---------------------------------------------------------------------
// assignClassToSelected: optimistic label then rollback on failure
// ---------------------------------------------------------------------

// Stryker flags dropping the `.filter((c) => ids.includes(c.id))` on
// `priors` in `assignClassToSelected` (clusterController.svelte.ts:149,
// MethodExpression mutant) as a surviving mutant. This is an accepted
// equivalent, not a coverage gap: `priors` is captured before the
// optimistic-label loop runs, so for any crop NOT in `ids`, the "prior"
// object is the exact same reference already sitting in `cropPager.items`
// — reverting it (`revertLocalLabel`) is a same-value no-op regardless of
// whether the filter ran. No test can observe the difference without
// spying on `cropPager.items`'s setter call count, which nothing else in
// this file does either.
describe('assignClassToSelected', () => {
  it('labels the selection, records undo off putCropLabel, and clears the selection on success (single crop)', async () => {
    vi.mocked(putCropLabel).mockResolvedValue({} as never);
    vi.spyOn(classesStore, 'byId').mockReturnValue({ id: 3, name: 'widget_a' } as never);
    const recordWritesSpy = vi.spyOn(undoStore, 'recordWrites');
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const { cropPager, sel, controller } = setup([a]);
    sel.ids = new Set(['a']);

    await controller.assignClassToSelected(3);

    expect(cropPager.items[0]!.class_id).toBe(3);
    expect(cropPager.items[0]!.class_validated).toBe(true);
    expect(cropPager.items[0]!.label_validated).toBe(true);
    expect(cropPager.items[0]!.label_source).toBe('human_confirmed');
    expect(putCropLabel).toHaveBeenCalledWith('a', 3);
    expect(recordWritesSpy).toHaveBeenCalledWith(['a']);
    expect(sel.size).toBe(0);
    expect(successSpy).toHaveBeenCalledWith('Labeled 1 crop.');
  });

  it('uses bulkLabel, follows the job and records the operation for undo for a multi-crop selection', async () => {
    mockLabelOp(bulkLabel, { updated: 2, conflicts: 0 });
    vi.spyOn(classesStore, 'byId').mockReturnValue({ id: 3, name: 'widget_a' } as never);
    const recordSpy = vi.spyOn(undoStore, 'recordLabelOp');
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { sel, controller } = setup([a, b]);
    sel.ids = new Set(['a', 'b']);

    await controller.assignClassToSelected(3);

    expect(bulkLabel).toHaveBeenCalledWith(['a', 'b'], 3);
    expect(recordSpy).toHaveBeenCalledWith('op1', ['a', 'b']);
    // Plural wording -- kills the ids.length === 1 ternary collapsing to
    // always-singular (a single-crop success elsewhere already covers
    // the singular branch producing the same string either way).
    expect(successSpy).toHaveBeenCalledWith('Labeled 2 crops.');
  });

  it("re-reads the served first page after a successful assign, so the footer count is the server's", async () => {
    mockLabelOp(bulkLabel, { updated: 2, conflicts: 0 });
    vi.spyOn(classesStore, 'byId').mockReturnValue({ id: 3, name: 'widget_a' } as never);
    const { sel, controller, loadFirst } = setup([crop('a'), crop('b')]);
    sel.ids = new Set(['a', 'b']);

    await controller.assignClassToSelected(3);

    expect(loadFirst).toHaveBeenCalledTimes(1);
  });

  it('does not re-read the page when the assign fails', async () => {
    vi.mocked(bulkLabel).mockRejectedValue(new Error('network down'));
    vi.spyOn(classesStore, 'byId').mockReturnValue({ id: 3, name: 'widget_a' } as never);
    vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const { sel, controller, loadFirst } = setup([crop('a'), crop('b')]);
    sel.ids = new Set(['a', 'b']);

    await controller.assignClassToSelected(3);

    expect(loadFirst).not.toHaveBeenCalled();
  });

  it('rolls back every optimistically-labeled crop when the write fails', async () => {
    vi.mocked(bulkLabel).mockRejectedValue(new Error('network down'));
    vi.spyOn(classesStore, 'byId').mockReturnValue({ id: 3, name: 'widget_a' } as never);
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a', { class_id: 1, class_name: 'widget_e' });
    const b = crop('b', { class_id: 1, class_name: 'widget_e' });
    const { cropPager, sel, controller } = setup([a, b]);
    sel.ids = new Set(['a', 'b']);

    await controller.assignClassToSelected(3);

    expect(cropPager.items.map((c) => c.class_id)).toEqual([1, 1]);
    expect(cropPager.items.map((c) => c.class_name)).toEqual(['widget_e', 'widget_e']);
    expect(errorSpy).toHaveBeenCalledWith('Label failed: network down');
  });

  it('warns and does nothing when nothing is selected', async () => {
    const warnSpy = vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const { controller } = setup([]);

    await controller.assignClassToSelected(3);

    expect(warnSpy).toHaveBeenCalledWith('Nothing selected.');
    expect(putCropLabel).not.toHaveBeenCalled();
  });

  it('errors and does nothing when the class id is unknown', async () => {
    vi.spyOn(classesStore, 'byId').mockReturnValue(undefined);
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a');
    const { sel, controller } = setup([a]);
    sel.ids = new Set(['a']);

    await controller.assignClassToSelected(999);

    expect(errorSpy).toHaveBeenCalledWith('Unknown class id 999');
    expect(putCropLabel).not.toHaveBeenCalled();
  });

  // Bug 4 (live smoke: /clusters/[id] header stayed "0 validated" after
  // labeling 34 crops through the page, only updating on reload). The
  // header chip reads `classesStore`'s server-owned `validated_count` —
  // a successful write must re-fetch it rather than leave it stale.
  it('refreshes classesStore after a successful label write so the header count is current', async () => {
    vi.mocked(putCropLabel).mockResolvedValue({} as never);
    vi.spyOn(classesStore, 'byId').mockReturnValue({ id: 3, name: 'widget_a' } as never);
    const refreshSpy = vi.spyOn(classesStore, 'refresh').mockResolvedValue();
    const a = crop('a');
    const { sel, controller } = setup([a]);
    sel.ids = new Set(['a']);

    await controller.assignClassToSelected(3);

    expect(refreshSpy).toHaveBeenCalledTimes(1);
  });

  it('does NOT refresh classesStore when the label write fails', async () => {
    vi.mocked(putCropLabel).mockRejectedValue(new Error('boom'));
    vi.spyOn(classesStore, 'byId').mockReturnValue({ id: 3, name: 'widget_a' } as never);
    vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const refreshSpy = vi.spyOn(classesStore, 'refresh').mockResolvedValue();
    const a = crop('a');
    const { sel, controller } = setup([a]);
    sel.ids = new Set(['a']);

    await controller.assignClassToSelected(3);

    expect(refreshSpy).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------
// handleClassDrop (drop-on-class / class-hotkey path) + excludedCropIds
// claim/release
// ---------------------------------------------------------------------

describe('handleClassDrop', () => {
  const cls: RegistryClass = { id: 7, name: 'truck' } as RegistryClass;

  it('prefers dragIds over droppedIds and the selection, and clears dragIds after reading it', async () => {
    mockLabelOp(bulkLabel, { updated: 1, conflicts: 0 });
    const a = crop('a');
    const b = crop('b');
    const { sel, controller, setDragIds, getDragIds } = setup([a, b]);
    setDragIds(['a']);
    sel.ids = new Set(['b']); // selection would pick 'b' -- dragIds must win

    await controller.handleClassDrop(cls, ['b']);

    expect(bulkLabel).toHaveBeenCalledWith(['a'], 7);
    expect(getDragIds()).toEqual([]);
  });

  it('falls back to droppedIds when no drag is in flight', async () => {
    mockLabelOp(bulkLabel, { updated: 1, conflicts: 0 });
    const b = crop('b');
    const { controller } = setup([b]);

    await controller.handleClassDrop(cls, ['b']);

    expect(bulkLabel).toHaveBeenCalledWith(['b'], 7);
  });

  it('falls back to the selection when there is no drag and no dropped id (class-hotkey path)', async () => {
    mockLabelOp(bulkLabel, { updated: 1, conflicts: 0 });
    const a = crop('a');
    const { sel, controller } = setup([a]);
    sel.ids = new Set(['a']);

    await controller.handleClassDrop(cls, []);

    expect(bulkLabel).toHaveBeenCalledWith(['a'], 7);
  });

  it('warns and makes no request when there is nothing to label', async () => {
    const warnSpy = vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const { controller } = setup([]);

    await controller.handleClassDrop(cls, []);

    expect(warnSpy).toHaveBeenCalledWith(
      'Select or drag crops first, then press a class hotkey.',
    );
    expect(bulkLabel).not.toHaveBeenCalled();
  });

  it('resets the grid override before the request settles and keeps the dropped crops in place', async () => {
    let resolveBulk!: (v: unknown) => void;
    vi.mocked(bulkLabel).mockReturnValue(
      new Promise((resolve) => {
        resolveBulk = resolve;
      }) as never,
    );
    const a = crop('a');
    const { cropPager, exclusionGuard, resetGrid, controller } = setup([a]);

    const p = controller.handleClassDrop(cls, ['a']);
    // Synchronously (before the request resolves) the drag-local grid
    // override is dropped; a label does not move the crop out of its
    // cluster, so it stays in the grid and is not claimed by the guard.
    expect(resetGrid).toHaveBeenCalledTimes(1);
    expect(cropPager.items.map((c) => c.id)).toEqual(['a']);
    expect(exclusionGuard.accept(a)).toBe(true);
    mockLabelOp(bulkLabel, { updated: 1 });
    resolveBulk({ op_id: 'op1', job_id: 'job1', status: 'queued' });
    await p;
  });

  it('labels the dropped crop in place: new class shown, total unchanged', async () => {
    mockLabelOp(bulkLabel, { updated: 1 });
    const a = crop('a');
    const b = crop('b');
    const { cropPager, controller } = setup([a, b]);
    cropPager.total = 2;

    await controller.handleClassDrop(cls, ['a']);

    expect(cropPager.items.map((c) => c.id)).toEqual(['a', 'b']);
    expect(cropPager.items[0]!.class_id).toBe(7);
    expect(cropPager.items[0]!.class_name).toBe('truck');
    expect(cropPager.items[0]!.class_validated).toBe(true);
    expect(cropPager.items[1]!.class_id).toBeNull();
    expect(cropPager.total).toBe(2);
  });

  it('rolls the drop back and keeps the crop when the label write fails', async () => {
    vi.mocked(bulkLabel).mockRejectedValue(new Error('network down'));
    vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a', { class_id: 1, class_name: 'widget_e' });
    const { cropPager, controller } = setup([a]);

    await controller.handleClassDrop(cls, ['a']);

    expect(cropPager.items.map((c) => c.class_id)).toEqual([1]);
  });

  it('records the operation for undo and toasts the plain success message on a clean (no-conflict) label', async () => {
    mockLabelOp(bulkLabel, { updated: 1 });
    const recordSpy = vi.spyOn(undoStore, 'recordLabelOp');
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const warnSpy = vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const a = crop('a');
    const { loadFirst, controller } = setup([a]);

    await controller.handleClassDrop(cls, ['a']);

    expect(recordSpy).toHaveBeenCalledWith('op1', ['a']);
    expect(successSpy).toHaveBeenCalledWith('Labeled 1 → truck.');
    expect(warnSpy).not.toHaveBeenCalled();
    expect(loadFirst).not.toHaveBeenCalled();
  });

  it('reloads and warns with the served counts when a worker blocked some of the drop', async () => {
    mockLabelOp(bulkLabel, { updated: 1, conflicts: 1 });
    const warnSpy = vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { loadFirst, controller } = setup([a, b]);

    await controller.handleClassDrop(cls, ['a', 'b']);

    expect(loadFirst).toHaveBeenCalledTimes(1);
    expect(warnSpy).toHaveBeenCalledWith(
      'Labeled 1 of 2 → truck (1 blocked by worker). Reloading.',
    );
  });
});
