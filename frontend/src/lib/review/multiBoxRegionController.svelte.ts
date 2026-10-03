/**
 * W8 multi-box region editing controller (docs/design/
 * w8-multibox-frontend-plan-2026-09-26.md) — owns the working `EditableBox[]`
 * set for the active crop's region slot and every write path over it,
 * following the existing factory-function convention
 * (`reviewController.svelte.ts`, `slotGalleryController.svelte.ts`):
 * `/review/+page.svelte` owns `queue`/`current`/`cursor` and hands them to
 * this controller by reference/accessor, rather than this file reaching
 * into the page directly. Extracted as its own module — not piled into
 * the already-3000-line page — per the same "extract a controller"
 * convention `reviewController.svelte.ts` set.
 *
 * Write policy (thin frontend — every write is immediate, no Save button,
 * matching the rest of this app):
 * - Per-box accept/reject (`y`/`r`) PATCHes that one box immediately
 *   (`PATCH /crops/{id}/regions/{box_id}`) and re-seeds from the server's
 *   returned item — it does NOT remove the crop from the queue, since the
 *   operator is typically still working through sibling boxes.
 * - Geometry edits (add/move/delete) stay purely local until the next
 *   confirm or explicit flush — sent as one `PUT /crops/{id}/regions` so
 *   an add+move+delete session becomes one write, one undo step (W8.8).
 * - Enter (`confirmAndAdvance`) is the owner-decided semantic: flips only
 *   `proposed` boxes to `accepted` (siblings' states are never touched),
 *   sends the accumulated geometry diff in the SAME PUT with
 *   `region_status: 'detected'`, and only then advances/removes the crop
 *   from the queue — matching every other slot-tab confirm action.
 * - Every successful write calls `undoStore.recordRegionWrites([cropId])`
 *   — Z's existing generic `queueController.undoLast()` already routes a
 *   `kind: 'region'` entry through `POST /crops/{id}/region/undo`, which
 *   the backend confirmed restores the whole prior `region_boxes` list
 *   (+ `region_status`/`region_revision`) in one step, so nothing new is
 *   needed on the undo side.
 */
import {
  toEditableBoxes,
  nextBoxIndex,
  removeBoxAt,
  addBox as addBoxPure,
  buildRegionsPutBoxes,
  confirmProposedBoxes,
  type EditableBox,
} from '$lib/annotations/multiBox';
import { putRegionBoxes, patchRegionBox, regionConflictDetail, ApiError } from '$lib/api';
import { slotOf } from '$lib/annotations/cropSlots';
import type { SlotSpec, BBoxNormLike } from '$lib/annotations/types';
import type { Crop } from '$lib/types';
import type { VectorRefresh } from '$lib/types_itemFilter';
import { undoStore } from '$stores/undo.svelte';
import { toastStore } from '$stores/toast.svelte';

export interface MultiBoxRegionController {
  readonly boxes: EditableBox[];
  readonly selectedIndex: number | null;
  readonly busy: boolean;
  /** True once a geometry edit (add/move/delete) hasn't been flushed to
   *  the server yet — drives an "unsaved edits" affordance if the caller
   *  wants one. */
  readonly dirty: boolean;
  seedFrom(crop: Crop | null): void;
  select(index: number): void;
  next(): void;
  addBox(box: BBoxNormLike): void;
  moveSelected(box: BBoxNormLike): void;
  deleteSelected(): void;
  /** y — accept the selected box. Immediate PATCH for a stored box; a
   *  local-only flip for a not-yet-saved new box (boxId === null). */
  acceptSelected(cropId: string): Promise<void>;
  /** r — reject the selected box. Same immediate-vs-local split as
   *  acceptSelected. */
  rejectSelected(cropId: string): Promise<void>;
  /** Enter — confirm: settle every `proposed` box to `accepted`, send the
   *  accumulated geometry diff in the same write, and report whether the
   *  caller should advance the queue (false on a server-side rejection,
   *  e.g. 422 no_accepted_box, so the crop stays in view for another
   *  per-box decision). */
  confirmAndSave(cropId: string): Promise<{ ok: boolean; item: Crop | null }>;
  /** Plain "Save" (no confirm semantics) — sends the accumulated geometry/
   *  delete/add diff as one `PUT /crops/{id}/regions`, with no
   *  `region_status` key, so every box's own state is preserved exactly
   *  as edited. Used by the standalone bbox-editor modal
   *  (`SlotBboxEditor.svelte`), which has no "confirm proposed" concept —
   *  unlike `confirmAndSave`, a no-op edit (nothing dirty, same box
   *  count) still round-trips through the server so `onsave` always
   *  fires with a real item. */
  saveEdits(cropId: string): Promise<{ ok: boolean; item: Crop | null }>;
  /** Sets the selected stored box's text (`PATCH .../regions/{box_id}`).
   *  A not-yet-saved local box has no id to address, so it is refused. */
  setSelectedText(cropId: string, text: string | null): Promise<void>;
  /** The served `region_profile.limits.max_boxes_per_write` for the
   *  active slot, or `null` when the slot declares none (a non-region
   *  slot). Never a client guess. */
  readonly maxBoxes: number | null;
  /** The item's served `region_revision`, echoed back as
   *  `expected_region_revision` on every write. */
  readonly revision: number | null;
  /** The last write's served `vector_refresh`: `pending` boxes have no
   *  vector yet. `null` before any write, after the next item is seeded,
   *  and when a write served none. */
  readonly vectorRefresh: VectorRefresh | null;
}

export interface MultiBoxRegionOptions {
  /** Called with every item the server returns (a successful write, or
   *  the current item inside a `region_conflict`), so the owner can keep
   *  its own copy of the crop in step. */
  onitem?: (crop: Crop) => void;
}

export function createMultiBoxRegionController(
  slot: () => SlotSpec | null,
  opts: MultiBoxRegionOptions = {},
): MultiBoxRegionController {
  let revision = $state<number | null>(null);
  let boxes = $state<EditableBox[]>([]);
  let original = $state<EditableBox[]>([]);
  let selectedIndex = $state<number | null>(null);
  let busy = $state(false);
  let vectorRefresh = $state<VectorRefresh | null>(null);
  // The page re-seeds this controller whenever its copy of the current item
  // is replaced (including by this controller's own write), so a re-seed of
  // the SAME crop must not drop what that write just served.
  let seededCropId: string | null = null;
  const dirty = $derived(boxes.some((b) => b.dirty) || boxes.length !== original.length);

  function seedFrom(crop: Crop | null): void {
    const s = slot();
    const data = crop && s ? slotOf(crop, s) : null;
    const editable = toEditableBoxes(data?.subBoxes ?? []);
    boxes = editable;
    original = editable;
    revision = data?.boxSet?.revision ?? null;
    selectedIndex = editable.length > 0 ? 0 : null;
    if ((crop?.id ?? null) !== seededCropId) vectorRefresh = null;
    seededCropId = crop?.id ?? null;
  }

  function select(index: number): void {
    if (index >= 0 && index < boxes.length) selectedIndex = index;
  }

  function next(): void {
    selectedIndex = nextBoxIndex(boxes.length, selectedIndex);
  }

  function addBox(box: BBoxNormLike): void {
    const result = addBoxPure(boxes, box);
    boxes = result.boxes;
    selectedIndex = result.selected;
  }

  function moveSelected(box: BBoxNormLike): void {
    if (selectedIndex == null) return;
    const idx = selectedIndex;
    boxes = boxes.map((b, i) => (i === idx ? { ...b, box, dirty: true } : b));
  }

  function deleteSelected(): void {
    if (selectedIndex == null) return;
    const result = removeBoxAt(boxes, selectedIndex);
    boxes = result.boxes;
    selectedIndex = result.selected;
  }

  function reseedFromWrittenCrop(crop: Crop): void {
    const s = slot();
    const data = s ? slotOf(crop, s) : null;
    const editable = toEditableBoxes(data?.subBoxes ?? []);
    boxes = editable;
    original = editable;
    revision = data?.boxSet?.revision ?? null;
    if (selectedIndex != null && selectedIndex >= editable.length) {
      selectedIndex = editable.length > 0 ? editable.length - 1 : null;
    }
    opts.onitem?.(crop);
  }

  /** A stale revision: adopt the server's current item and tell the
   *  operator, rather than guessing which of their edits still apply. */
  function adoptConflict(e: unknown): Crop | null {
    const conflict = regionConflictDetail(e);
    if (!conflict) return null;
    reseedFromWrittenCrop(conflict.item);
    toastStore.warn(
      'This item changed since it was loaded; showing the latest boxes. Repeat your change if it still applies.',
    );
    return conflict.item;
  }

  async function flipSelected(
    cropId: string,
    state: 'accepted' | 'rejected',
  ): Promise<void> {
    if (selectedIndex == null || busy) return;
    const box = boxes[selectedIndex];
    if (box.boxId == null) {
      // Not-yet-saved local box: nothing to PATCH, just flip it in place.
      const idx = selectedIndex;
      boxes = boxes.map((b, i) => (i === idx ? { ...b, state } : b));
      return;
    }
    busy = true;
    try {
      const { crop, vectorRefresh: vr } = await patchRegionBox(cropId, box.boxId, {
        state,
        expectedRegionRevision: revision ?? undefined,
      });
      vectorRefresh = vr;
      reseedFromWrittenCrop(crop);
      undoStore.recordRegionWrites([cropId]);
    } catch (e) {
      if (adoptConflict(e)) return;
      const msg = e instanceof ApiError ? e.message : (e as Error).message;
      toastStore.error(
        `Box ${state === 'accepted' ? 'accept' : 'reject'} failed: ${msg}`,
      );
    } finally {
      busy = false;
    }
  }

  async function acceptSelected(cropId: string): Promise<void> {
    await flipSelected(cropId, 'accepted');
  }

  async function rejectSelected(cropId: string): Promise<void> {
    await flipSelected(cropId, 'rejected');
  }

  async function confirmAndSave(
    cropId: string,
  ): Promise<{ ok: boolean; item: Crop | null }> {
    if (busy) return { ok: false, item: null };
    busy = true;
    try {
      const confirmed = confirmProposedBoxes(boxes);
      const body = buildRegionsPutBoxes(original, confirmed);
      const { crop, vectorRefresh: vr } = await putRegionBoxes(cropId, body, {
        regionStatus: 'detected',
        expectedRegionRevision: revision ?? undefined,
      });
      vectorRefresh = vr;
      reseedFromWrittenCrop(crop);
      undoStore.recordRegionWrites([cropId]);
      return { ok: true, item: crop };
    } catch (e) {
      const adopted = adoptConflict(e);
      if (adopted) return { ok: false, item: adopted };
      const msg = e instanceof ApiError ? e.message : (e as Error).message;
      toastStore.error(`Confirm failed: ${msg}`);
      return { ok: false, item: null };
    } finally {
      busy = false;
    }
  }

  async function saveEdits(cropId: string): Promise<{ ok: boolean; item: Crop | null }> {
    if (busy) return { ok: false, item: null };
    busy = true;
    try {
      const body = buildRegionsPutBoxes(original, boxes);
      const { crop, vectorRefresh: vr } = await putRegionBoxes(cropId, body, {
        expectedRegionRevision: revision ?? undefined,
      });
      vectorRefresh = vr;
      reseedFromWrittenCrop(crop);
      undoStore.recordRegionWrites([cropId]);
      return { ok: true, item: crop };
    } catch (e) {
      const adopted = adoptConflict(e);
      if (adopted) return { ok: false, item: adopted };
      const msg = e instanceof ApiError ? e.message : (e as Error).message;
      toastStore.error(`Save failed: ${msg}`);
      return { ok: false, item: null };
    } finally {
      busy = false;
    }
  }

  async function setSelectedText(cropId: string, text: string | null): Promise<void> {
    if (selectedIndex == null || busy) return;
    const box = boxes[selectedIndex];
    if (box.boxId == null) {
      toastStore.warn('Save the new box first; a reading is written per saved box.');
      return;
    }
    busy = true;
    try {
      const { crop, vectorRefresh: vr } = await patchRegionBox(cropId, box.boxId, {
        text,
        expectedRegionRevision: revision ?? undefined,
      });
      vectorRefresh = vr;
      reseedFromWrittenCrop(crop);
      undoStore.recordRegionWrites([cropId]);
    } catch (e) {
      if (adoptConflict(e)) return;
      const msg = e instanceof ApiError ? e.message : (e as Error).message;
      toastStore.error(`Text save failed: ${msg}`);
    } finally {
      busy = false;
    }
  }

  return {
    get boxes() {
      return boxes;
    },
    get selectedIndex() {
      return selectedIndex;
    },
    get busy() {
      return busy;
    },
    get dirty() {
      return dirty;
    },
    get maxBoxes() {
      return slot()?.capabilities.subBox?.maxBoxesPerWrite ?? null;
    },
    get revision() {
      return revision;
    },
    get vectorRefresh() {
      return vectorRefresh;
    },
    seedFrom,
    select,
    next,
    addBox,
    moveSelected,
    deleteSelected,
    acceptSelected,
    rejectSelected,
    confirmAndSave,
    saveEdits,
    setSelectedText,
  };
}
