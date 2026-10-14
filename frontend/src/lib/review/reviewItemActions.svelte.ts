/**
 * `/review` item actions on a non-slot tab: assign a class, confirm the
 * proposal, accept the model's class, skip, discard and undo, plus the
 * quick-assign class strip they draw from. Thin wrappers over the queue
 * action controller (`reviewController.svelte.ts`), which owns the
 * optimistic remove and the rollback.
 *
 * Moved verbatim out of `review/+page.svelte`.
 */

import {
  itemHintClassIds,
  quickAssignClasses,
  resolveConfirmClassId,
} from '$lib/classPicker';
import { probeOpinion } from '$lib/review/probeOpinion';
import { kg } from '$lib/review/reviewKeys';
import type { ReviewQueueLoader } from '$lib/review/reviewQueueLoader.svelte';
import type { ReviewItem } from '$lib/types_review';
import { classesStore } from '$stores/classes.svelte';
import { toastStore } from '$stores/toast.svelte';

export function createReviewItemActions(loader: ReviewQueueLoader) {
  const { queueController } = loader;
  const current = $derived(loader.current);

  // R1 (visual audit 2026-09-24): item-class targets only — a slot-bound
  // region class is never an item's class — ranked by what THIS item
  // already points at (proposal / current / VLM / model), then by
  // validated count, instead of by global validated count alone.
  const currentHintIds = $derived(itemHintClassIds(current));
  const topClasses = $derived(
    quickAssignClasses(classesStore.classes, currentHintIds, 10),
  );

  // P1-5: what Enter/Confirm would actually assign, or null when there's
  // nothing to confirm (67/100 `all`-tab items today). Drives the Confirm
  // button's disabled state and whether Enter confirms vs. opens the class
  // picker.
  const canConfirm = $derived(resolveConfirmClassId(current) != null);

  /**
   * Optimistically drop an item from the queue and advance. Thin wrapper
   * around the controller's `removeFromQueue` — kept under this name
   * because slot actions (confirmSlot/rejectSlot/markFalsePositive)
   * still call it directly.
   */
  function _removeFromQueue(item: ReviewItem): () => void {
    return queueController.removeFromQueue(item);
  }

  async function assign(classId: number): Promise<void> {
    if (!current) return;
    await queueController.assign(current, classId);
  }

  async function confirmAndAdvance(): Promise<void> {
    if (!current) return;
    const proposed = resolveConfirmClassId(current);
    if (proposed == null) {
      // Enter's registration already routes here vs. openPicker()
      // based on canConfirm, so this only fires from the Confirm button —
      // which is disabled in this state — or a stale click race. Keep the
      // toast as a safety net either way.
      toastStore.warn(
        `No proposed class on this item — press ${kg('review.queue.class_picker')} to search.`,
      );
      return;
    }
    await assign(proposed);
  }

  /** "Accept model's class" (item 14, 2026-09-24 logic-moves): the
   *  model_disagreements tab's probe prediction now carries its own
   *  `probe_pred_class_id`, so this assigns it directly — no name→id
   *  lookup needed (closes G4). */
  // F8 D1: the served probe opinion decides whether a prediction is
  // shown and whether "Accept model's class" is offered.
  const opinion = $derived(
    current ? probeOpinion(current) : { kind: 'none' as const, showAccept: false },
  );

  async function acceptModelClass(): Promise<void> {
    if (!current || current.probe_pred_class_id == null || !opinion.showAccept) return;
    await assign(current.probe_pred_class_id);
  }

  function skip(): void {
    queueController.skip();
  }

  async function discard(): Promise<void> {
    if (!current) return;
    // Discard = "permanently dismiss this crop from every review queue."
    // Stamps review_dismissed_at on the backend; the review queue's
    // must_not filter excludes any crop with that field set. The crop's
    // class / region state is left intact — this is NOT an unlabel. See
    // reviewController.svelte.ts's `discard` for the undo-entry note.
    await queueController.discard(current);
  }

  async function undoLast(): Promise<void> {
    // `crop` (from POST {API_PREFIX}/crops/{id}/label/undo or
    // .../label/undo_batch) already carries its own served
    // proposed_class_id/_name (item 11, 2026-09-24 logic-moves) — no
    // client fill-in. `reason` is the only field the controller adds;
    // every other queue-only field has no meaningful value for a
    // restored item.
    await queueController.undoLast();
  }

  return {
    get currentHintIds() {
      return currentHintIds;
    },
    get topClasses() {
      return topClasses;
    },
    get canConfirm() {
      return canConfirm;
    },
    _removeFromQueue,
    assign,
    confirmAndAdvance,
    get opinion() {
      return opinion;
    },
    acceptModelClass,
    skip,
    discard,
    undoLast,
  };
}

export type ReviewItemActions = ReturnType<typeof createReviewItemActions>;
