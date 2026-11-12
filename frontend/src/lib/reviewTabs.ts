/**
 * The Review view's queue-capable slots. The queues themselves (To label, Check machine labels,
 * Disagreements, New class proposals, Audit sample) are served by `GET /review/tabs` and carry no
 * frontend structure; only a slot's queue (the region queue) needs a tab here, because the slot
 * brings its own panel, keymap and edit mode.
 */
import { registeredSlots, onRegisteredSlotsChanged } from './annotations/registeredSlots';
import type { SlotKey, SlotSpec } from './annotations/types';
import type { ReviewTab, SlotReviewTab } from '$lib/types_review';

export interface ReviewTabDef {
  id: ReviewTab;
  label: string;
  /** The slot's `queue.urlId`, its bookmark value. */
  urlId: string;
  /** `{API_PREFIX}/review/{endpointId}`: the served queue id the slot reads. */
  endpointId: string;
  slot: SlotSpec;
}

/**
 * The internal tab id of a slot is the structural `slot:${key}` template, not its `urlId`, so a
 * second queue-capable slot gets an independent tab with zero edits here.
 */
export function slotTabId(key: SlotKey): SlotReviewTab {
  return `slot:${key}`;
}

export function buildReviewTabs(slots: SlotSpec[]): ReviewTabDef[] {
  return slots
    .filter((s) => s.capabilities.queue)
    .map((s) => {
      const q = s.capabilities.queue!;
      return {
        id: slotTabId(s.key),
        label: q.tabLabel,
        urlId: q.urlId,
        endpointId: q.endpointId,
        slot: s,
      };
    });
}

/** `let`, not `const`: rebuilt in place when a deployment profile installs more slots, before
 *  first render (see `annotations/registeredSlots.ts`). */
export let REVIEW_TABS: ReviewTabDef[] = buildReviewTabs(registeredSlots);

onRegisteredSlotsChanged(() => {
  REVIEW_TABS = buildReviewTabs(registeredSlots);
});

/** True for any tab backed by a slot's queue capability; structural, so it stays correct for a
 *  tab id whose slot has since been unregistered. */
export function isSlotTab(id: ReviewTab): id is SlotReviewTab {
  return id.startsWith('slot:');
}

/** The internal tab of a served queue id: the slot's tab when a slot reads that queue, else the
 *  queue id itself (a plain queue has no slot). */
export function tabForQueue(queue: string | null): ReviewTab {
  return REVIEW_TABS.find((t) => t.endpointId === queue)?.id ?? queue ?? '';
}

/** `{API_PREFIX}/review/{endpointId}`: what `getReviewQueue` should call for a tab. */
export function endpointForTab(
  id: ReviewTab,
  tabs: ReviewTabDef[] = REVIEW_TABS,
): string {
  return tabs.find((t) => t.id === id)?.endpointId ?? id;
}
