/**
 * Pure logic for W8 multi-box region edits — box selection/geometry
 * helpers and the `PUT /crops/{crop_id}/regions` request body builder
 * (spec: openprocessor any_domain_plan.md §7.7, W8.7/W8.8). Kept dependency-
 * free (no api.ts, no Svelte) so it can be unit-tested in isolation and
 * mutation-checked without a component mount.
 *
 * See docs/design/w8-multibox-frontend-plan-2026-09-26.md.
 */

import type { SlotBox, BBoxNormLike, XYXY } from './types';

/** A box as edited in the UI, before it is sent to the server. Mirrors
 *  `SlotBox` but geometry is always the crop-local `parent` frame (what
 *  the editor draws in) and `boxId: null` marks a newly-drawn box. */
export interface EditableBox {
  boxId: string | null;
  state: string;
  /** Crop-local (`parent`-frame) geometry. Null only transiently. */
  box: BBoxNormLike | null;
  /** True once the geometry has been moved/resized/created by this edit
   *  session, vs. loaded unchanged from the server. Drives whether the
   *  PUT body sends `bbox_norm` for this element at all. */
  dirty: boolean;
}

/** Converts served `SlotBox`es (source + server-projected parent frame)
 *  into the editor's working set. A box with no `parent` projection
 *  (`bboxInParentField` absent/null) is dropped — spec: "null → not
 *  drawable in the crop view". */
export function toEditableBoxes(boxes: SlotBox[]): EditableBox[] {
  return boxes
    .filter((b) => b.parent != null)
    .map((b) => ({ boxId: b.boxId, state: b.state, box: b.parent, dirty: false }));
}

/** Cycles the selected index forward through `boxes` (Tab / `box_edit.next_box`).
 *  Wraps around; returns `null` when there are no boxes to select. */
export function nextBoxIndex(count: number, current: number | null): number | null {
  if (count <= 0) return null;
  if (current == null) return 0;
  return (current + 1) % count;
}

/** Removes the box at `index` (Backspace/Delete on the selected box) and
 *  returns the new list plus the selection that should follow it — the
 *  same index (now pointing at the next box), clamped, or `null` when
 *  the list is now empty. */
export function removeBoxAt(
  boxes: EditableBox[],
  index: number,
): { boxes: EditableBox[]; selected: number | null } {
  if (index < 0 || index >= boxes.length) return { boxes, selected: index };
  const next = boxes.slice(0, index).concat(boxes.slice(index + 1));
  const selected = next.length === 0 ? null : Math.min(index, next.length - 1);
  return { boxes: next, selected };
}

/** Appends a newly-drawn box (owner decision: unbounded — no client cap;
 *  only the served `limits.max_boxes_per_write` disables Add, enforced by
 *  the caller, not here). Returns the new list and the new box's index. */
export function addBox(
  boxes: EditableBox[],
  box: BBoxNormLike,
  state = 'accepted',
): { boxes: EditableBox[]; selected: number } {
  const next = boxes.concat([{ boxId: null, state, box, dirty: true }]);
  return { boxes: next, selected: next.length - 1 };
}

/** One element of `PUT /crops/{crop_id}/regions` `boxes` (W8.8). */
export type RegionBoxInput =
  | { box_id: string }
  | { box_id: string; bbox_norm: XYXY }
  | { box_id: string; bbox_norm: XYXY; state: string }
  | { box_id: string; state: string }
  | { box_id: null; bbox_norm: XYXY; state?: string };

function toXyxy(b: BBoxNormLike): XYXY {
  return [b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2];
}

/**
 * Builds the `boxes` array for `PUT /crops/{crop_id}/regions` from the
 * editor's current working set against the originally-loaded boxes.
 *
 * Per W8.8: an untouched stored box is sent as `{box_id}` alone (keeps
 * geometry/state/reason/text verbatim); a moved one as `{box_id,
 * bbox_norm}` (keeps state); a box whose `state` was flipped includes
 * `state`; a brand-new box is `{box_id: null, bbox_norm}` with no `state`
 * (relies on the server's `new_box_default: "accepted"` for the PUT
 * routes — plan ambiguity #2). A stored box absent from `edited` is
 * omitted, which the server treats as a delete.
 */
export function buildRegionsPutBoxes(
  original: EditableBox[],
  edited: EditableBox[],
): RegionBoxInput[] {
  const originalById = new Map(
    original.filter((b) => b.boxId != null).map((b) => [b.boxId, b]),
  );
  return edited.map((b): RegionBoxInput => {
    if (b.boxId == null) {
      if (!b.box) throw new Error('a new box requires geometry');
      return { box_id: null, bbox_norm: toXyxy(b.box) };
    }
    const orig = originalById.get(b.boxId);
    const stateChanged = orig != null && orig.state !== b.state;
    const geometryChanged = b.dirty && b.box != null;
    if (!geometryChanged && !stateChanged) return { box_id: b.boxId };
    if (geometryChanged && stateChanged) {
      return {
        box_id: b.boxId,
        bbox_norm: toXyxy(b.box as BBoxNormLike),
        state: b.state,
      };
    }
    if (geometryChanged)
      return { box_id: b.boxId, bbox_norm: toXyxy(b.box as BBoxNormLike) };
    return { box_id: b.boxId, state: b.state };
  });
}

/**
 * Owner decision (binding): Enter confirms only `proposed` boxes.
 * `rejected`/`false_positive` boxes are left exactly as they are — a
 * whole-set confirm never overrides a per-box decision (W8.7).
 * Returns the edited set with every `proposed` box flipped to `accepted`,
 * for building the confirm-write's boxes; does NOT itself decide whether
 * the result has an accepted box (see `hasAcceptedBox`) — the server is
 * the one that raises `no_accepted_box`, this is only what the client
 * requests.
 */
export function confirmProposedBoxes(boxes: EditableBox[]): EditableBox[] {
  return boxes.map((b) => (b.state === 'proposed' ? { ...b, state: 'accepted' } : b));
}

/** True when at least one box in the set is (or would be) `accepted` —
 *  used only for a client-side pre-check / disabled-state hint before
 *  sending Enter; the server's `no_accepted_box` 422 is still the
 *  authority (thin-frontend rule — this never blocks a request the
 *  server would actually accept, it only pre-empts a request the server
 *  is guaranteed to reject given the boxes as currently known). */
export function hasAcceptedBox(boxes: EditableBox[]): boolean {
  return boxes.some((b) => b.state === 'accepted');
}
