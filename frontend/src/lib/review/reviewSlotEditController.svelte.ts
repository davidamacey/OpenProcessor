/**
 * `/review` slot-tab editing state: the multi-box working set, edit mode,
 * the inline slot metadata editors (reading, status, rejection reason) and
 * their per-crop saves, the box ring styling, and the reseed that runs on
 * every cursor move.
 *
 * Moved verbatim out of `review/+page.svelte`.
 */

import { untrack } from 'svelte';
import { apiErrorText, patchSlotMeta } from '$lib/api';
import { slotOf } from '$lib/annotations/cropSlots';
import { AbortRegistry } from '$lib/review/abortRegistry';
import { capCropDisplayStyle } from '$lib/review/cropDisplaySize';
import { createMultiBoxRegionController } from '$lib/review/multiBoxRegionController.svelte';
import type { ReviewQueueLoader } from '$lib/review/reviewQueueLoader.svelte';
import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
import { humanWritableStates, panelLabels } from '$lib/review/slotPanel';
import type { ReviewItem } from '$lib/types';
import { regionStatusesStore, toneRingRgb } from '$stores/regionStatuses.svelte';
import { toastStore } from '$stores/toast.svelte';

export interface ReviewSlotEditOptions {
  tabs: ReviewTabsController;
  loader: ReviewQueueLoader;
}

export function createReviewSlotEditController(opts: ReviewSlotEditOptions) {
  const { queue } = opts.loader;
  const activeSlot = $derived(opts.tabs.activeSlot);
  const current = $derived(opts.loader.current);

  // Multi-box regions (docs/design/w8-multibox-frontend-plan-2026-09-26.md):
  // the controller owns the working box set; every item the server returns
  // (a write, or the current item inside a revision conflict) patches the
  // queue's own copy so the panel never reads a stale one.
  const multiBox = createMultiBoxRegionController(() => activeSlot, {
    onitem: (crop) => {
      const idx = queue.items.findIndex((x) => x.id === crop.id);
      if (idx >= 0) queue.items[idx] = { ...queue.items[idx], ...crop } as ReviewItem;
    },
  });

  // A slot that declares `listField` (the served region slot, always) uses
  // MultiBoxCanvas/multiBoxRegionController. A tier-2 slot with only a
  // scalar `bboxField` is display-only: the backend has no write route for
  // it, so it has no edit mode.
  const isMultiBoxSlot = $derived(activeSlot?.capabilities.subBox?.listField != null);

  // -- slot-tab actions -------------------------------------------------
  // The canvas is always live for a multi-box slot (`multiBox` owns the
  // working set): tweak boxes, accept/reject one, and Enter confirms the
  // proposed ones. The goal is one keystroke (Enter) per item when
  // scanning thousands of crops.
  // DQ-M5: natural pixel size of the currently-rendered crop thumbnail,
  // read back via Svelte's bind:naturalWidth/naturalHeight once the <img>
  // loads. Drives capCropDisplayStyle() so a tiny crop upscales by at most
  // CROP_UPSCALE_CAP instead of filling the whole (now height-capped)
  // panel — see cropDisplaySize.ts for the full rationale.
  let cropNaturalWidth = $state(0);
  let cropNaturalHeight = $state(0);
  const cropDisplayStyle = $derived(
    capCropDisplayStyle(cropNaturalWidth, cropNaturalHeight),
  );
  let slotCanvas = $state<{ handleKey: (e: KeyboardEvent) => boolean } | null>(null);
  // Read-only by default: the canvas only becomes interactive when the
  // operator presses E (or clicks Edit bbox). Most cascade-detected
  // proposals are already correct — forcing the heavy drag-handle UI on
  // every crop is what made the tab feel "weird" vs. the other review
  // tabs. Edit mode resets to false on every cursor advance so the
  // operator always lands on the next item in scan-and-confirm mode.
  let editMode = $state<boolean>(false);

  // Inline editors for the slot metadata fields. Seeded from the
  // current crop's SlotData on every cursor advance; saved on blur /
  // Enter via patchSlotMeta (PATCH {slot's own patchMeta endpoint}).
  // Each field saves independently with optimistic-UI + revert-on-error,
  // matching the assign() pattern.
  let editedSlotText = $state<string>('');
  let editedSlotStatus = $state<string>('');
  let editedRejectionReason = $state<string>('');

  // Status values an operator is allowed to write, for the ACTIVE slot —
  // closes Finding D (the panel used to render one fixed slot's
  // vocabulary regardless of which slot tab was active). Order matches
  // the deployment's served `GET {API_PREFIX}/regions/statuses` vocabulary
  // when loaded, falling back to the active slot's own
  // `capabilities.lifecycle.states`.
  const slotStatusOptions = $derived(
    activeSlot ? humanWritableStates(activeSlot, regionStatusesStore.list) : [],
  );
  const slotLabels = $derived(activeSlot ? panelLabels(activeSlot) : null);

  // Multi-box: per-box state → ring color/dash. The ring color
  // reads the served box_states `tone` via `toneRingRgb(boxStateTone(state))` —
  // `boxStateTone` returns 'neutral' on a pre-tone backend or an
  // unrecognized state, so this never invents a color the server didn't
  // choose.
  function multiBoxRingColor(state: string): string {
    return toneRingRgb(regionStatusesStore.boxStateTone(state));
  }
  function multiBoxDashed(state: string): boolean {
    return (
      regionStatusesStore.boxStateInfo(state)?.dashed ??
      (state === 'rejected' || state === 'false_positive')
    );
  }
  function multiBoxStateLabel(state: string): string {
    const served = regionStatusesStore.boxStateInfo(state)?.label;
    if (served) return served;
    if (state === 'accepted') return 'accepted';
    if (state === 'proposed') return 'awaiting verification';
    if (state === 'false_positive') return 'false positive';
    if (state === 'rejected') return 'rejected';
    return state;
  }

  // Reseed whenever the cursor changes (advancing to next crop) or the
  // tab/items reset. Also exit edit mode so the next item lands in
  // read-only scan mode regardless of where we left the previous one.
  //
  // The ONLY dependency is the current crop's id: everything after that
  // runs untracked, so a drag tick or nudge never reseeds the working
  // set from the server snapshot and drops edit mode.
  $effect(() => {
    const id = current?.id;
    untrack(() => reseedForCrop(id ?? null));
  });

  /** The served box the operator has selected (a stored box; a new local
   *  box has no served data yet). */
  const selectedSlotBox = $derived.by(() => {
    if (!current || !activeSlot || !isMultiBoxSlot) return null;
    const id = multiBox.boxes[multiBox.selectedIndex ?? -1]?.boxId ?? null;
    if (id == null) return null;
    return slotOf(current, activeSlot)?.subBoxes?.find((b) => b.boxId === id) ?? null;
  });

  /** The text the reading input starts from: the selected box's reading
   *  for a multi-box slot, the item-level value for a scalar-box slot. */
  function seededText(): string {
    if (!current || !activeSlot) return '';
    if (isMultiBoxSlot) return selectedSlotBox?.text ?? '';
    return slotOf(current, activeSlot)?.text?.value ?? '';
  }

  $effect(() => {
    const text = seededText();
    untrack(() => {
      editedSlotText = text;
    });
  });

  function reseedForCrop(id: string | null): void {
    // DQ-M5: drop the previous crop's natural size immediately so its cap
    // never briefly applies to the next crop's <img> before it loads and
    // rebinds naturalWidth/naturalHeight.
    cropNaturalWidth = 0;
    cropNaturalHeight = 0;
    const seedData = current && activeSlot ? slotOf(current, activeSlot) : null;
    editedSlotText = seededText();
    editedSlotStatus = seedData?.lifecycle?.status ?? '';
    editedRejectionReason = seedData?.lifecycle?.rejectionReason ?? '';
    editMode = false;
    if (isMultiBoxSlot) multiBox.seedFrom(current ?? null);
    void id;
  }

  // In-flight slot-meta saves, keyed by crop id so concurrent edits to
  // the same crop are aborted-then-replaced (the latest blur wins) and
  // edits to a *different* crop don't interfere with each other.
  const slotMetaAborts = new AbortRegistry();

  /**
   * PATCH a slot metadata field and render whatever the server returns —
   * no client-computed post-write state. On failure, reseed the local
   * inputs from the crop's last-known-good server state (rather than a
   * hand-rolled "prior" snapshot) so the operator sees what's actually
   * persisted.
   */
  async function saveSlotMeta(patch: {
    status?: string | null;
    text?: string | null;
    rejectionReason?: string | null;
  }): Promise<void> {
    if (!current || !activeSlot) return;
    const id = current.id;
    // Look up by id, not cursor — if the user advances mid-save the
    // captured idx would point at the next crop and a reseed would
    // corrupt unrelated state.
    const findIdx = () => queue.items.findIndex((x) => x.id === id);
    // Abort any in-flight save on this crop so we don't get an ABA-style
    // response that overwrites a newer edit.
    const ac = slotMetaAborts.start(id);
    try {
      const res = await patchSlotMeta(activeSlot, id, patch, ac.signal);
      const idx = findIdx();
      if (idx >= 0) {
        queue.items[idx] = { ...queue.items[idx], ...res.item } as ReviewItem;
      }
    } catch (e) {
      if (ac.signal.aborted) return; // superseded by a newer save
      // Reseed local inputs only if we're still on the same crop the
      // user was editing; otherwise leave the inputs alone — they're
      // already bound to the new crop's state.
      if (current?.id === id) {
        const idx = findIdx();
        const seedData = idx >= 0 ? slotOf(queue.items[idx], activeSlot) : null;
        editedSlotText = seededText();
        editedSlotStatus = seedData?.lifecycle?.status ?? '';
        editedRejectionReason = seedData?.lifecycle?.rejectionReason ?? '';
      }
      toastStore.error(`Save failed: ${apiErrorText(e)}`);
    } finally {
      slotMetaAborts.finish(id, ac);
    }
  }

  async function commitSlotText(): Promise<void> {
    if (!current || !activeSlot) return;
    const next = editedSlotText.trim() || null;
    if (isMultiBoxSlot) {
      // A reading is per box: PATCH the selected stored box.
      if ((selectedSlotBox?.text ?? null) === next) return;
      await multiBox.setSelectedText(current.id, next);
      return;
    }
    const slotData = slotOf(current, activeSlot);
    if ((slotData?.text?.value ?? null) === next) return;
    await saveSlotMeta({ text: next });
  }

  async function commitSlotStatus(): Promise<void> {
    if (!current || !activeSlot) return;
    if (!editedSlotStatus) return;
    const slotData = slotOf(current, activeSlot);
    if (editedSlotStatus === slotData?.lifecycle?.status) return;
    // The server decides what a status does to the box list (a
    // `clears_box` status empties it); the returned item is rendered as is.
    await saveSlotMeta({ status: editedSlotStatus });
  }

  async function commitRejectionReason(): Promise<void> {
    if (!current || !activeSlot) return;
    const slotData = slotOf(current, activeSlot);
    const next = editedRejectionReason.trim() || null;
    if ((slotData?.lifecycle?.rejectionReason ?? null) === next) return;
    await saveSlotMeta({ rejectionReason: next });
  }

  function toggleEdit(): void {
    // Only a multi-box slot has an edit mode; a scalar-box slot is
    // display-only (the backend has no write route for it).
    if (!current || !isMultiBoxSlot) return;
    if (editMode) {
      // Cancel-style exit: drop local edits and reseed from server state.
      multiBox.seedFrom(current);
      editMode = false;
      return;
    }
    editMode = true;
  }

  return {
    multiBox,
    get isMultiBoxSlot() {
      return isMultiBoxSlot;
    },
    get cropNaturalWidth() {
      return cropNaturalWidth;
    },
    set cropNaturalWidth(v: number) {
      cropNaturalWidth = v;
    },
    get cropNaturalHeight() {
      return cropNaturalHeight;
    },
    set cropNaturalHeight(v: number) {
      cropNaturalHeight = v;
    },
    get cropDisplayStyle() {
      return cropDisplayStyle;
    },
    get slotCanvas() {
      return slotCanvas;
    },
    set slotCanvas(v: { handleKey: (e: KeyboardEvent) => boolean } | null) {
      slotCanvas = v;
    },
    get editMode() {
      return editMode;
    },
    get editedSlotText() {
      return editedSlotText;
    },
    set editedSlotText(v: string) {
      editedSlotText = v;
    },
    get editedSlotStatus() {
      return editedSlotStatus;
    },
    set editedSlotStatus(v: string) {
      editedSlotStatus = v;
    },
    get editedRejectionReason() {
      return editedRejectionReason;
    },
    set editedRejectionReason(v: string) {
      editedRejectionReason = v;
    },
    get slotStatusOptions() {
      return slotStatusOptions;
    },
    get slotLabels() {
      return slotLabels;
    },
    multiBoxRingColor,
    multiBoxDashed,
    multiBoxStateLabel,
    get selectedSlotBox() {
      return selectedSlotBox;
    },
    commitSlotText,
    commitSlotStatus,
    commitRejectionReason,
    toggleEdit,
  };
}

export type ReviewSlotEditController = ReturnType<typeof createReviewSlotEditController>;
