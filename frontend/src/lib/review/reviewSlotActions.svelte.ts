/**
 * `/review` slot-tab writes: confirm, reject (with the in-app reason
 * prompt), false positive, the multi-box confirm, and the local step-back
 * stack that re-queues the last actioned crop.
 *
 * Moved verbatim out of `review/+page.svelte`.
 */

import { apiErrorText, getCrop, patchSlotMeta } from '$lib/api';
import type { ReviewItemActions } from '$lib/review/reviewItemActions.svelte';
import { kg } from '$lib/review/reviewKeys';
import type { ReviewQueueLoader } from '$lib/review/reviewQueueLoader.svelte';
import type { ReviewSlotEditController } from '$lib/review/reviewSlotEditController.svelte';
import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
import { statusWantsRejectionReason } from '$lib/review/slotPanel';
import { pushUndo, removeUndo, popUndo, reinsertAt } from '$lib/review/slotQueueOps';
import type { ReviewItem } from '$lib/types';
import { regionStatusesStore } from '$stores/regionStatuses.svelte';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';

export interface ReviewSlotActionsOptions {
  tabs: ReviewTabsController;
  loader: ReviewQueueLoader;
  actions: ReviewItemActions;
  edit: ReviewSlotEditController;
}

export function createReviewSlotActions(opts: ReviewSlotActionsOptions) {
  const { loader } = opts;
  const { queue, handledIds } = loader;
  const { _removeFromQueue } = opts.actions;
  const { multiBox } = opts.edit;
  const activeSlot = $derived(opts.tabs.activeSlot);
  const current = $derived(loader.current);

  // DQ-m6 (docs/design/data-quality-pass-2026-09-24.md): rejectSlot()
  // used `window.prompt()` for this — a native, OS-level dialog. It
  // blocks the JS thread while open, which is exactly why the audit's
  // screenshot/automation pass saw "no prompt before or after the
  // write": a native dialog renders outside the page's DOM/CDP surface,
  // so nothing shows up in a page screenshot, and an automated
  // click/keypress driver that doesn't specifically arm a native-dialog
  // handler gets it silently auto-dismissed (Playwright's default),
  // which reads as "the prompt didn't appear" even though the code path
  // ran. An in-app modal is real DOM — screenshot-visible, keyboard-
  // driveable the same way every other modal on this page already is
  // (Enter submits, Esc cancels), and testable without special dialog
  // plumbing.
  let rejectReasonPromptOpen = $state(false);
  let rejectReasonPromptValue = $state('');
  let rejectReasonPromptResolve: ((value: string | null) => void) | null = null;

  function promptForRejectionReason(): Promise<string | null> {
    rejectReasonPromptValue = '';
    rejectReasonPromptOpen = true;
    return new Promise((resolve) => {
      rejectReasonPromptResolve = resolve;
    });
  }

  function submitRejectReasonPrompt(): void {
    rejectReasonPromptOpen = false;
    rejectReasonPromptResolve?.(rejectReasonPromptValue.trim() || null);
    rejectReasonPromptResolve = null;
  }

  function cancelRejectReasonPrompt(): void {
    rejectReasonPromptOpen = false;
    rejectReasonPromptResolve?.(null);
    rejectReasonPromptResolve = null;
  }

  const undoHint = () =>
    `Press ${kg('review.undo')} to undo, step back with ${kg('review.region.back')}.`;

  // Undo stack for slot confirm/reject. Each entry holds the previously
  // confirmed box so "Back" can re-insert the crop into the queue and
  // restore what the user just saved (allowing them to fix a mistake
  // without re-finding the crop). Bounded to 20 entries — enough for
  // half a session of confusion, small enough to keep memory tiny.
  interface SlotUndoEntry {
    item: ReviewItem;
    insertAt: number;
  }
  // $state.raw, not $state: deep reactivity would proxy every pushed entry,
  // so _removeSlotUndo could never match the raw object the caller holds.
  // Every mutation reassigns the array, so raw is just as reactive here.
  let slotUndoStack = $state.raw<SlotUndoEntry[]>([]);
  const SLOT_UNDO_MAX = 20;
  function _pushSlotUndo(entry: SlotUndoEntry): void {
    slotUndoStack = pushUndo(slotUndoStack, entry, SLOT_UNDO_MAX);
  }

  /** Drop a specific step-back entry — used when its API call failed. */
  function _removeSlotUndo(entry: SlotUndoEntry): void {
    slotUndoStack = removeUndo(slotUndoStack, entry);
  }

  async function slotBack(): Promise<void> {
    const { entry: last, rest } = popUndo(slotUndoStack);
    if (!last) {
      toastStore.info('Nothing to step back to.');
      return;
    }
    slotUndoStack = rest;
    // Back in play: let loadMore surface it again if a later page returns it.
    handledIds.delete(last.item.id);
    // Refetch the crop so the operator sees what the database actually
    // holds — the local snapshot can lag (e.g. another worker re-ran
    // OCR or another curator edited concurrently). This is the
    // "confidence in changes" guarantee the user asked for.
    let fresh: ReviewItem;
    try {
      const c = await getCrop(last.item.id);
      // The {API_PREFIX}/crops/{id} endpoint returns a Crop, but the review
      // queue carries extra fields (reason, proposed_*). Keep the
      // snapshot's queue-only metadata and overlay the authoritative
      // store fields (including the freshly re-mapped .slots) on top.
      fresh = { ...last.item, ...c } as ReviewItem;
    } catch (e) {
      toastStore.warn(`Re-fetch failed; restoring local snapshot: ${apiErrorText(e)}`);
      fresh = last.item;
    }
    const insertAt = Math.min(last.insertAt, queue.items.length);
    queue.items = reinsertAt(queue.items, last.insertAt, fresh);
    queue.total = queue.total + 1;
    loader.cursor = insertAt;
    toastStore.info(
      `Stepped back. Press ${kg('review.region.edit_box')} to re-edit, ${kg('review.region.confirm')} to re-confirm.`,
    );
  }

  /** Confirm for a slot with no box list: a status-only write. */
  async function confirmSlot(): Promise<void> {
    if (!current || !activeSlot) return;
    const confirmStatus =
      regionStatusesStore.confirmStatus ??
      activeSlot.capabilities.lifecycle?.confirmState;
    if (!confirmStatus) return;
    const item = current;
    const undoEntry: SlotUndoEntry = { item, insertAt: loader.cursor };
    _pushSlotUndo(undoEntry);
    const restore = _removeFromQueue(item);
    try {
      await patchSlotMeta(activeSlot, item.id, { status: confirmStatus });
      undoStore.recordRegionWrites([item.id]);
      toastStore.success(`${activeSlot.label.title} confirmed. ${undoHint()}`);
    } catch (e) {
      _removeSlotUndo(undoEntry);
      restore();
      toastStore.error(`Confirm failed: ${apiErrorText(e)}`);
    }
  }

  /**
   * Multi-box Enter: confirms only `proposed` boxes,
   * leaving `rejected`/`false_positive` siblings exactly as they are, in
   * the same `PUT /crops/{id}/regions` write as any pending geometry edit
   * (add/move/delete) — see `multiBoxRegionController.confirmAndSave`.
   * Only advances the queue on success; a server rejection (e.g. 422
   * `no_accepted_box`) leaves the crop in view so the operator can accept
   * a box first (`y`) and press Enter again. Z (undo) is unaffected —
   * `confirmAndSave` already calls `undoStore.recordRegionWrites`, so the
   * existing `queueController.undoLast()` restores the whole prior list
   * via `POST /crops/{id}/region/undo` (backend-confirmed one-step
   * restore, see the plan doc's "resolved" section).
   */
  async function confirmMultiBoxSlot(): Promise<void> {
    if (!current || !activeSlot) return;
    const item = current;
    const restore = _removeFromQueue(item);
    const { ok } = await multiBox.confirmAndSave(item.id);
    if (ok) {
      toastStore.success(`${activeSlot.label.title} confirmed. ${undoHint()}`);
    } else {
      restore();
    }
  }

  async function rejectSlot(): Promise<void> {
    if (!current || !activeSlot) return;
    const item = current;
    // m5 (2026-09-24 interactive pass): reject used to clear the box with
    // no chance to record why, even when the served reject status has
    // `wants_reason:true` (e.g. `no_region_visible`) — the rejection-
    // reason input only ever showed up after the fact, when the status
    // dropdown had already caught up to the write. Ask up front instead,
    // using the served vocabulary to decide whether to ask at all.
    const rejectStatus =
      regionStatusesStore.rejectStatus ?? activeSlot.capabilities.lifecycle?.rejectState;
    if (!rejectStatus) return;
    let reason: string | null = null;
    if (statusWantsRejectionReason(activeSlot, rejectStatus, regionStatusesStore.list)) {
      reason = await promptForRejectionReason();
    }
    const undoEntry: SlotUndoEntry = { item, insertAt: loader.cursor };
    _pushSlotUndo(undoEntry);
    const restore = _removeFromQueue(item);
    try {
      // One write, so one Z undoes it: the served reject status clears the
      // box list server-side and carries the reason in the same request.
      await patchSlotMeta(activeSlot, item.id, {
        status: rejectStatus,
        ...(reason ? { rejectionReason: reason } : {}),
      });
      undoStore.recordRegionWrites([item.id]);
      toastStore.success(`${activeSlot.label.title} rejected. ${undoHint()}`);
    } catch (e) {
      _removeSlotUndo(undoEntry);
      restore();
      toastStore.error(`Reject failed: ${apiErrorText(e)}`);
    }
  }

  async function markFalsePositive(): Promise<void> {
    if (!current || !activeSlot) return;
    const fpState =
      regionStatusesStore.falsePositiveStatus ??
      activeSlot.capabilities.lifecycle?.falsePositiveState;
    if (!fpState) return; // no falsePositiveState declared -> action shouldn't be reachable
    const item = current;
    // False positive: a detector drew this box but it is NOT the slot's
    // subject. We KEEP the box + all detection metadata (unlike Reject,
    // which clears it) — flipping only status. The retained geometry
    // feeds FP analysis and becomes a hard negative in the dedicated
    // training export.
    const undoEntry: SlotUndoEntry = { item, insertAt: loader.cursor };
    _pushSlotUndo(undoEntry);
    const restore = _removeFromQueue(item);
    try {
      await patchSlotMeta(activeSlot, item.id, { status: fpState });
      undoStore.recordRegionWrites([item.id]);
      toastStore.success(`Marked false positive (box kept). ${undoHint()}`);
    } catch (e) {
      _removeSlotUndo(undoEntry);
      restore();
      toastStore.error(`Mark FP failed: ${apiErrorText(e)}`);
    }
  }

  return {
    get rejectReasonPromptOpen() {
      return rejectReasonPromptOpen;
    },
    get rejectReasonPromptValue() {
      return rejectReasonPromptValue;
    },
    set rejectReasonPromptValue(v: string) {
      rejectReasonPromptValue = v;
    },
    submitRejectReasonPrompt,
    cancelRejectReasonPrompt,
    get slotUndoStack() {
      return slotUndoStack;
    },
    slotBack,
    confirmSlot,
    confirmMultiBoxSlot,
    rejectSlot,
    markFalsePositive,
  };
}

export type ReviewSlotActions = ReturnType<typeof createReviewSlotActions>;
