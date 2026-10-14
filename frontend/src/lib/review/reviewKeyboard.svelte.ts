/**
 * `/review` keyboard wiring: the `review` keyboard scope, the per-class
 * hotkey dispatch, and the tab-action keymap (queue actions on a core tab,
 * the slot keymap and per-box keys on a slot tab, the bbox canvas forward in
 * edit mode).
 *
 * Moved verbatim out of `review/+page.svelte`.
 */

import { buildSlotKeymap } from '$lib/review/slotKeymap';
import { isSlotSuppressedTab } from '$lib/review/slotTabGuard';
import type { ReviewClassPicker } from '$lib/review/reviewClassPicker.svelte';
import type { ReviewItemActions } from '$lib/review/reviewItemActions.svelte';
import type { ReviewQueueLoader } from '$lib/review/reviewQueueLoader.svelte';
import type { ReviewSlotActions } from '$lib/review/reviewSlotActions.svelte';
import type { ReviewSlotEditController } from '$lib/review/reviewSlotEditController.svelte';
import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
import { isSlotTab } from '$lib/reviewTabs';
import type { RegistryClass } from '$lib/types_classes';
import { dropOnClassStore } from '$stores/dropOnClass.svelte';
import { keyboardStore, type RegisterActionOptions } from '$stores/keyboard.svelte';
import { toastStore } from '$stores/toast.svelte';

export interface ReviewKeyboardOptions {
  tabs: ReviewTabsController;
  loader: ReviewQueueLoader;
  actions: ReviewItemActions;
  picker: ReviewClassPicker;
  edit: ReviewSlotEditController;
  slot: ReviewSlotActions;
}

export function installReviewKeyboard(opts: ReviewKeyboardOptions): void {
  const { tabs, loader, edit } = opts;
  const { queue, maybePrefetch } = loader;
  const { assign, confirmAndAdvance, discard, skip, undoLast } = opts.actions;
  const { openPicker } = opts.picker;
  const { multiBox, toggleEdit } = edit;
  const { confirmSlot, confirmMultiBoxSlot, rejectSlot, markFalsePositive, slotBack } =
    opts.slot;
  const tab = $derived(tabs.tab);
  const activeSlot = $derived(tabs.activeSlot);
  const awaitingDeepLink = $derived(tabs.awaitingDeepLink);
  const current = $derived(loader.current);
  const canConfirm = $derived(opts.actions.canConfirm);
  const editMode = $derived(edit.editMode);
  const isMultiBoxSlot = $derived(edit.isMultiBoxSlot);

  $effect(() => {
    keyboardStore.setScope('review');
  });

  // Per-class hotkeys defined on /classes are routed here through the
  // layout-level global keydown listener via dropOnClassStore. Pressing
  // a class's bound letter assigns the current crop and advances —
  // matching the cluster page's bulk-label dispatch shape so the same
  // hotkey works everywhere it makes sense. The regions tab is a
  // different flow (confirming a bbox, not a class) so we no-op there
  // and leave the letters free for region actions.
  $effect(() => {
    if (isSlotSuppressedTab(tab)) return;
    const off = dropOnClassStore.register(async (cls: RegistryClass) => {
      if (!current) {
        toastStore.info('No item to label.');
        return;
      }
      await assign(cls.id);
    });
    return () => off();
  });

  // Keyboard shortcuts. Per-class letter hotkeys (configured on /classes)
  // are routed through dropOnClassStore by the layout-level keydown
  // listener and work on every tab. The shortcuts below are the
  // tab-action shortcuts; on a slot tab Enter/D get rebound to the slot's
  // confirm/reject so the same finger pattern works for both flows.
  //
  // On a slot tab, behavior splits between read-only scan mode
  // (default) and edit mode (operator pressed E or Edit bbox):
  //   - read-only: arrows page the queue, Enter confirms-and-advances,
  //     E enters edit mode — matches the other review tabs.
  //   - edit:      arrows nudge the bbox, Enter saves+exits edit mode,
  //     Esc cancels edit, the bbox canvas owns the keystroke flow.
  $effect(() => {
    // DQ-M7: register nothing while a `?crop_id=` deep link is still
    // resolving — otherwise a keypress during that window (previously
    // ~4s, paging through the whole queue up to the target page) acts on
    // whatever item #1 of the just-loaded first page happens to be, not
    // the crop the operator followed the link to review.
    if (awaitingDeepLink) return;

    const offs: Array<() => void> = [];
    const reg = (
      actionId: string,
      fn: () => void | Promise<void>,
      opts?: RegisterActionOptions,
    ) =>
      offs.push(keyboardStore.registerAction(actionId, () => void fn(), 'review', opts));

    if (activeSlot) {
      // Table built by the slot-generic slotKeymap module (P2.8c), reading
      // activeSlot.capabilities.queue.keymap instead of a second
      // hand-maintained copy — asserted by slotKeymap.test.ts rather than
      // only readable here.
      for (const entry of buildSlotKeymap(activeSlot, editMode, {
        confirm: isMultiBoxSlot ? confirmMultiBoxSlot : confirmSlot,
        reject: rejectSlot,
        markFalsePositive,
        toggleEdit,
        back: slotBack,
        advance: () => {
          loader.cursor = Math.min(queue.items.length - 1, loader.cursor + 1);
          maybePrefetch();
        },
        // Enter in edit mode is bound to box_edit.save, not
        // review.region.confirm — for a multi-box slot both must run the
        // SAME confirm (owner decision: Enter always confirms, in scan or
        // edit mode, in one write).
        saveAndExit: confirmMultiBoxSlot,
      })) {
        reg(entry.actionId, entry.fn, {
          keys: [entry.combo],
          description: entry.description,
        });
      }
      if (isMultiBoxSlot && current) {
        // Per-box actions (y/r/Tab) — reserved letters, see
        // keymapFallback.ts. Available in both scan and edit mode, since
        // accept/reject/select don't require entering edit.
        const cropId = current.id;
        reg('review.region.accept_box', () => multiBox.acceptSelected(cropId));
        reg('review.region.reject_box', () => multiBox.rejectSelected(cropId));
        reg('box_edit.next_box', () => multiBox.next());
        if (editMode) {
          reg('box_edit.delete_box', () => multiBox.deleteSelected());
        }
      }
    } else {
      reg('review.queue.confirm', () => {
        // P1-5: a blank proposal made Enter a silent no-op. Open the
        // class picker instead so the operator can act in one keystroke
        // rather than hitting Enter and wondering why nothing happened.
        if (canConfirm) return confirmAndAdvance();
        openPicker();
      });
      reg('review.queue.discard', discard);
      reg('review.queue.class_picker', openPicker);
    }
    if (!editMode) {
      // In edit mode the queue never moves: N/Z (and the arrows, which the
      // canvas owns below) only act once the edit is saved or cancelled.
      reg('review.skip', skip);
      reg('review.undo', undoLast);
    }

    let canvasKey: ((e: KeyboardEvent) => void) | null = null;
    if (isMultiBoxSlot && editMode) {
      // Edit mode only: forward the `box_edit` nudge / right-edge / clear
      // keys into the slot's bbox canvas (it resolves them through the
      // keymap). Outside edit mode arrows page the queue like every other
      // tab.
      canvasKey = (e: KeyboardEvent) => {
        const slotCanvas = edit.slotCanvas;
        if (!slotCanvas) return;
        const target = e.target as HTMLElement | null;
        if (target && /^(input|textarea|select)$/i.test(target.tagName)) return;
        if (slotCanvas.handleKey(e)) e.preventDefault();
      };
      window.addEventListener('keydown', canvasKey);
    } else if (!isSlotTab(tab)) {
      // On non-slot tabs arrow keys navigate the queue.
      reg('review.queue.prev', () => {
        loader.cursor = Math.max(0, loader.cursor - 1);
      });
      reg('review.queue.next', () => {
        loader.cursor = Math.min(queue.items.length - 1, loader.cursor + 1);
        maybePrefetch();
      });
    }

    return () => {
      offs.forEach((off) => off());
      if (canvasKey) window.removeEventListener('keydown', canvasKey);
    };
  });
}
