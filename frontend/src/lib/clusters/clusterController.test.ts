/**
 * Behavior tests for the /clusters/[id] action controller
 * (docs/design/test-audit-2026-09-24.md P1-4, second pass). Replaces
 * `clusterMoveRace.test.ts`'s source-scan `describe('wiring: ... actually
 * uses the exclusion set', ...)` block — see that file's remaining header
 * comment for what's still covered there (the raw pager/gridGroups
 * mechanism) versus here (every call site that claims/releases against
 * the shared `ExclusionGuard`).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  createClusterActionController,
  createExclusionGuard,
} from './clusterController.svelte';
import { createPager } from '$lib/pager.svelte';
import { createSelection } from '$lib/selection.svelte';
import type { Crop, RegistryClass } from '$lib/types';

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

import {
  ApiError,
  bulkLabel,
  discardCrop,
  discardCropsBatch,
  excludeCrops,
  moveCropsToCluster,
  putCropLabel,
  unexcludeCrops,
  undoCropLabel,
  vlmDismissCrop,
} from '$lib/api';
import { classesStore } from '$stores/classes.svelte';
import { undoStore } from '$stores/undo.svelte';
import { toastStore } from '$stores/toast.svelte';

function crop(id: string, extra: Partial<Crop> = {}): Crop {
  return {
    id,
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
    class_id: null,
    class_name: null,
    class_source: null,
    label_source: 'model',
    label_validated: false,
    class_validated: false,
    label_confidence: null,
    cluster_id: 42,
    similarity_to_centroid: null,
    cluster_subid: null,
    ...extra,
  } as Crop;
}

function setup(items: Crop[]) {
  const cropPager = createPager<Crop>({
    fetchPage: async () => ({ items: [], total: 0 }),
    keyOf: (c) => c.id,
  });
  cropPager.items = items;
  cropPager.total = items.length;
  const sel = createSelection({ plainClick: 'replace' });
  const exclusionGuard = createExclusionGuard();
  let dragIds: string[] = [];
  const resetGrid = vi.fn();
  const loadFirst = vi.fn(async () => {});
  const rememberTarget = vi.fn();
  let visible = items;
  const controller = createClusterActionController({
    cropPager,
    sel,
    exclusionGuard,
    resetGrid,
    getDragIds: () => dragIds,
    setDragIds: (ids) => {
      dragIds = ids;
    },
    getVisibleCrops: () => visible,
    getClusterId: () => 42,
    rememberTarget,
    loadFirst,
  });
  return {
    cropPager,
    sel,
    exclusionGuard,
    controller,
    resetGrid,
    loadFirst,
    rememberTarget,
    getDragIds: () => dragIds,
    setDragIds: (ids: string[]) => {
      dragIds = ids;
    },
    setVisible: (v: Crop[]) => {
      visible = v;
    },
  };
}

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
    vi.spyOn(classesStore, 'byId').mockReturnValue({ id: 3, name: 'sedan' } as never);
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

  it('uses bulkLabel and records undo off the server updated_ids for a multi-crop selection', async () => {
    vi.mocked(bulkLabel).mockResolvedValue({
      updated: 2,
      updated_ids: ['a', 'b'],
      conflicts: [],
    });
    vi.spyOn(classesStore, 'byId').mockReturnValue({ id: 3, name: 'sedan' } as never);
    const recordWritesSpy = vi.spyOn(undoStore, 'recordWrites');
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { sel, controller } = setup([a, b]);
    sel.ids = new Set(['a', 'b']);

    await controller.assignClassToSelected(3);

    expect(bulkLabel).toHaveBeenCalledWith(['a', 'b'], 3);
    expect(recordWritesSpy).toHaveBeenCalledWith(['a', 'b']);
    // Plural wording -- kills the ids.length === 1 ternary collapsing to
    // always-singular (a single-crop success elsewhere already covers
    // the singular branch producing the same string either way).
    expect(successSpy).toHaveBeenCalledWith('Labeled 2 crops.');
  });

  it('rolls back every optimistically-labeled crop when the write fails', async () => {
    vi.mocked(bulkLabel).mockRejectedValue(new Error('network down'));
    vi.spyOn(classesStore, 'byId').mockReturnValue({ id: 3, name: 'sedan' } as never);
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a', { class_id: 1, class_name: 'coupe' });
    const b = crop('b', { class_id: 1, class_name: 'coupe' });
    const { cropPager, sel, controller } = setup([a, b]);
    sel.ids = new Set(['a', 'b']);

    await controller.assignClassToSelected(3);

    expect(cropPager.items.map((c) => c.class_id)).toEqual([1, 1]);
    expect(cropPager.items.map((c) => c.class_name)).toEqual(['coupe', 'coupe']);
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
});

// ---------------------------------------------------------------------
// handleClassDrop (drop-on-class / class-hotkey path) + excludedCropIds
// claim/release
// ---------------------------------------------------------------------

describe('handleClassDrop', () => {
  const cls: RegistryClass = { id: 7, name: 'truck' } as RegistryClass;

  it('prefers dragIds over droppedIds and the selection, and clears dragIds after reading it', async () => {
    vi.mocked(bulkLabel).mockResolvedValue({
      updated: 1,
      updated_ids: ['a'],
      conflicts: [],
    });
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
    vi.mocked(bulkLabel).mockResolvedValue({
      updated: 1,
      updated_ids: ['b'],
      conflicts: [],
    });
    const b = crop('b');
    const { controller } = setup([b]);

    await controller.handleClassDrop(cls, ['b']);

    expect(bulkLabel).toHaveBeenCalledWith(['b'], 7);
  });

  it('falls back to the selection when there is no drag and no dropped id (class-hotkey path)', async () => {
    vi.mocked(bulkLabel).mockResolvedValue({
      updated: 1,
      updated_ids: ['a'],
      conflicts: [],
    });
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

  it('claims dropped ids in the exclusion guard before the request settles, and resets the grid', async () => {
    let resolveBulk!: (v: {
      updated: number;
      updated_ids: string[];
      conflicts: never[];
    }) => void;
    vi.mocked(bulkLabel).mockReturnValue(
      new Promise((resolve) => {
        resolveBulk = resolve;
      }),
    );
    const a = crop('a');
    const { exclusionGuard, resetGrid, controller } = setup([a]);

    const p = controller.handleClassDrop(cls, ['a']);
    // Synchronously (before the request resolves) the id must already be
    // claimed and the grid override dropped -- this is what closes the
    // stale-fetch race a concurrent GET could otherwise win.
    expect(exclusionGuard.accept(a)).toBe(false);
    expect(resetGrid).toHaveBeenCalledTimes(1);
    resolveBulk({ updated: 1, updated_ids: ['a'], conflicts: [] });
    await p;
  });

  it('removes only the dropped crop from the grid (not the whole list) and decrements total by exactly ids.length', async () => {
    vi.mocked(bulkLabel).mockResolvedValue({
      updated: 1,
      updated_ids: ['a'],
      conflicts: [],
    });
    const a = crop('a');
    const b = crop('b');
    const { cropPager, controller } = setup([a, b]);
    cropPager.total = 1; // deliberately desynced from items.length (2)

    await controller.handleClassDrop(cls, ['a']);

    expect(cropPager.items.map((c) => c.id)).toEqual(['b']);
    // Math.max(0, 1 - 1) === 0 -- without the clamp (or with -/+ flipped)
    // this would go negative or increase instead.
    expect(cropPager.total).toBe(0);
  });

  it('decrements total by exactly ids.length in the ordinary (well above zero) case, not clamping it to 0', async () => {
    vi.mocked(bulkLabel).mockResolvedValue({
      updated: 1,
      updated_ids: ['a'],
      conflicts: [],
    });
    const a = crop('a');
    const { cropPager, controller } = setup([a]);
    cropPager.total = 5; // well above ids.length -- Math.max(0, x) === x here,
    // so Math.min(0, x) (a survived mutant) would wrongly clamp to 0.

    await controller.handleClassDrop(cls, ['a']);

    expect(cropPager.total).toBe(4);
  });

  it('records undo off the server updated_ids and toasts the plain success message on a clean (no-conflict) label', async () => {
    vi.mocked(bulkLabel).mockResolvedValue({
      updated: undefined as unknown as number,
      updated_ids: ['a'],
      conflicts: [],
    });
    const recordWritesSpy = vi.spyOn(undoStore, 'recordWrites');
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const warnSpy = vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const a = crop('a');
    const { loadFirst, controller } = setup([a]);

    await controller.handleClassDrop(cls, ['a']);

    expect(recordWritesSpy).toHaveBeenCalledWith(['a']);
    // res.updated ?? ids.length -- res.updated is undefined here, so the
    // fallback (ids.length === 1) must be used.
    expect(successSpy).toHaveBeenCalledWith('Labeled 1 → truck.');
    expect(warnSpy).not.toHaveBeenCalled();
    expect(loadFirst).not.toHaveBeenCalled();
  });

  it('treats a missing `conflicts` array as zero conflicts rather than throwing', async () => {
    vi.mocked(bulkLabel).mockResolvedValue({
      updated: 1,
      updated_ids: ['a'],
      conflicts: undefined as unknown as never[],
    });
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const { controller } = setup([a]);

    await expect(controller.handleClassDrop(cls, ['a'])).resolves.toBeUndefined();

    expect(successSpy).toHaveBeenCalledWith('Labeled 1 → truck.');
  });

  it('releases conflicted ids (they never actually left) and reloads instead of toasting plain success', async () => {
    vi.mocked(bulkLabel).mockResolvedValue({
      updated: 1,
      updated_ids: ['a'],
      conflicts: [{ crop_id: 'b', current_source: 'vlm' }],
    });
    const warnSpy = vi.spyOn(toastStore, 'warn').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { exclusionGuard, loadFirst, controller } = setup([a, b]);

    await controller.handleClassDrop(cls, ['a', 'b']);

    expect(exclusionGuard.accept(b)).toBe(true); // released
    expect(exclusionGuard.accept(a)).toBe(false); // stays claimed
    expect(loadFirst).toHaveBeenCalledTimes(1);
    expect(warnSpy).toHaveBeenCalledWith(
      'Labeled 1 of 2 → truck (1 blocked by worker). Reloading.',
    );
  });

  it('reverts the optimistic removal and releases claimed ids when the request fails outright', async () => {
    vi.mocked(bulkLabel).mockRejectedValue(new Error('boom'));
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a');
    const b = crop('b');
    const { cropPager, exclusionGuard, controller } = setup([a, b]);

    await controller.handleClassDrop(cls, ['a']);

    expect(cropPager.items.map((c) => c.id)).toEqual(['a', 'b']);
    expect(cropPager.total).toBe(2);
    expect(exclusionGuard.accept(a)).toBe(true);
    expect(errorSpy).toHaveBeenCalledWith('Label failed: boom');
  });
});

// ---------------------------------------------------------------------
// acceptVlmForCrop / rejectVlmForCrop
// ---------------------------------------------------------------------

describe('acceptVlmForCrop', () => {
  it('applies the suggestion optimistically and records undo on success', async () => {
    vi.mocked(putCropLabel).mockResolvedValue({} as never);
    const recordWritesSpy = vi.spyOn(undoStore, 'recordWrites');
    const a = crop('a', { vlm_suggested_class_id: 9, vlm_suggested_class_name: 'van' });
    const { cropPager, controller } = setup([a]);

    await controller.acceptVlmForCrop(a);

    expect(cropPager.items[0]!.class_id).toBe(9);
    // ?? null (not && null): a truthy name must survive, not collapse to
    // null.
    expect(cropPager.items[0]!.class_name).toBe('van');
    expect(putCropLabel).toHaveBeenCalledWith('a', 9);
    expect(recordWritesSpy).toHaveBeenCalledWith(['a']);
  });

  it('reverts the optimistic label when the write fails', async () => {
    vi.mocked(putCropLabel).mockRejectedValue(new Error('nope'));
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a', { vlm_suggested_class_id: 9, vlm_suggested_class_name: 'van' });
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
    vi.mocked(bulkLabel).mockImplementation(async (ids, _classId) => ({
      updated: ids.length,
      updated_ids: ids,
      conflicts: [],
    }));
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const recordWritesSpy = vi.spyOn(undoStore, 'recordWrites');
    const a = crop('a', { vlm_suggested_class_id: 1, vlm_suggested_class_name: 'x1' });
    const b = crop('b', { vlm_suggested_class_id: 1, vlm_suggested_class_name: 'x1' });
    const c = crop('c', { vlm_suggested_class_id: 2, vlm_suggested_class_name: 'x2' });
    const { cropPager, controller, setVisible } = setup([a, b, c]);
    setVisible([a, b, c]);

    await controller.acceptAllVlmOnPage();

    expect(bulkLabel).toHaveBeenCalledTimes(2);
    expect(bulkLabel).toHaveBeenCalledWith(['a', 'b'], 1);
    expect(bulkLabel).toHaveBeenCalledWith(['c'], 2);
    // recordWrites is called once per group, off that group's own
    // server-served updated_ids -- not skipped/dropped.
    expect(recordWritesSpy).toHaveBeenCalledWith(['a', 'b']);
    expect(recordWritesSpy).toHaveBeenCalledWith(['c']);
    expect(successSpy).toHaveBeenCalledWith('Accepted 3 suggestions.');
    // ?? null (not && null): a truthy suggested name must survive.
    expect(cropPager.items.find((x) => x.id === 'a')!.class_name).toBe('x1');
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
  it('rolls back only the crops in a failed group, leaving the succeeded group labeled', async () => {
    vi.mocked(bulkLabel).mockImplementation(async (ids, classId) => {
      if (classId === 2) throw new Error('group 2 failed');
      return { updated: ids.length, updated_ids: ids, conflicts: [] };
    });
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'x');
    const a = crop('a', { vlm_suggested_class_id: 1, vlm_suggested_class_name: 'x1' });
    const c = crop('c', { vlm_suggested_class_id: 2, vlm_suggested_class_name: 'x2' });
    const { cropPager, controller, setVisible } = setup([a, c]);
    setVisible([a, c]);

    await controller.acceptAllVlmOnPage();

    const byId = new Map(cropPager.items.map((x) => [x.id, x]));
    expect(byId.get('a')!.class_id).toBe(1); // succeeded group stays labeled
    expect(byId.get('c')!.class_id).toBeNull(); // failed group reverted
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
});

// ---------------------------------------------------------------------
// ignore / unignore
// ---------------------------------------------------------------------

describe('ignoreSelected / undoIgnore', () => {
  it('excludes only the selected crops from the grid (leaving an unselected crop in place), claims the exclusion guard, and remembers the batch', async () => {
    vi.mocked(excludeCrops).mockResolvedValue({ excluded: 1, errors: 0 });
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
    vi.mocked(excludeCrops).mockResolvedValue({ excluded: 1, errors: 0 });
    const successSpy = vi.spyOn(toastStore, 'success').mockImplementation(() => 'x');
    const a = crop('a');
    const { sel, controller } = setup([a]);
    sel.ids = new Set(['a']);

    await controller.ignoreSelected();

    expect(excludeCrops).toHaveBeenCalledWith(['a'], 'ignore');
    expect(successSpy).toHaveBeenCalledWith('Ignored 1. Press U to undo.');
  });

  it('undoIgnore restores the last-ignored batch, releases the exclusion guard, and toasts the exact restored count', async () => {
    vi.mocked(excludeCrops).mockResolvedValue({ excluded: 1, errors: 0 });
    vi.mocked(unexcludeCrops).mockResolvedValue({ unexcluded: 1, errors: 0 });
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
    vi.mocked(excludeCrops).mockResolvedValue({ excluded: 1, errors: 0 });
    vi.mocked(unexcludeCrops).mockResolvedValue({ unexcluded: 1, errors: 0 });
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
    vi.mocked(excludeCrops).mockResolvedValue({ excluded: 1, errors: 0 });
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
