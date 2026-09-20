import { registeredSlots, onRegisteredSlotsChanged } from './annotations/registeredSlots';
import type { SlotKey, SlotSpec } from './annotations/types';
import type { ReviewTab, SlotReviewTab } from './types';

export interface ReviewTabDef {
  id: ReviewTab;
  label: string;
  /** `?tab=` value. Equals `id` for core tabs; equals the slot's
   *  `queue.urlId` for slot tabs. Kept separate from `id` so a future
   *  internal-id scheme (e.g. `slot:${key}`) doesn't have to touch every
   *  existing bookmark — see the plan's §3.3 for the full design this is
   *  a deliberately-scoped-down slice of. */
  urlId: string;
  /** `{API_PREFIX}/review/{endpointId}`. Equals `id` for core tabs;
   *  equals the slot's `queue.endpointId` for slot tabs. */
  endpointId: string;
  /** Present only for slot tabs — the whole "is this a slot tab?" test. */
  slot?: SlotSpec;
}

/**
 * Top-level `/review` tabs (2026-09 tab consolidation). Down from 9 to 5.
 *
 * A review of all 9 tabs against the real 1,000-crop index found three
 * were too big to be curated queues — Mismatches (1,000 · 97.5% the size
 * of "All"), Gemma Low-Conf (1,000 · 11% of the dataset), and Primary ·
 * Low-Conf (1,000 · 92% of the ENTIRE dataset). Those three collapsed
 * into quick-filter preset chips shown on the `all` tab instead (see
 * REVIEW_PRESETS below) — same backend cohort query as before
 * (`/curation/review/{id}`, via resolveEffectiveTab), just triggered from a chip
 * instead of a nav tab.
 *
 * Outliers was retired entirely: 3 live rows, half its backend query
 * already dead code by its own comment, and functionally identical to the
 * `atypicality` sort already available via the strategy bar. Its backend
 * `/curation/review/outliers` endpoint is untouched (still reachable, just no
 * longer linked from this UI).
 *
 * Uncertainty, Model Disagreements, and COCO Blind Spots stay as
 * top-level tabs unchanged — each is a real, distinct signal the live
 * counts backed up (100 / 12 / 1,000 respectively).
 */
export const CORE_REVIEW_TABS: ReviewTabDef[] = [
  { id: 'all', label: 'All', urlId: 'all', endpointId: 'all' },
  {
    id: 'uncertainty',
    label: 'Uncertainty',
    urlId: 'uncertainty',
    endpointId: 'uncertainty',
  },
  // Phase 5 active-learning loop: validated crops where the newly
  // promoted model disagrees with the human label.
  {
    id: 'model_disagreements',
    label: 'Model Disagreements',
    urlId: 'model_disagreements',
    endpointId: 'model_disagreements',
  },
  // Primary-subject active-learning queue — COCO-confirmed vehicles v6
  // missed entirely. Genuinely distinct failure mode from the rest.
  {
    id: 'coco_blind_spots',
    label: 'COCO Blind Spots',
    urlId: 'coco_blind_spots',
    endpointId: 'coco_blind_spots',
  },
];

/**
 * Slot tabs — derived from each queue-capable slot's `QueueCapability`
 * (docs/genericization-plan-2026-09-13.md §3.3/P2.8, finished by the
 * §9.5 addendum) instead of a hand-maintained literal. The internal tab
 * id is the structural `slot:${key}` template (`slotTabId`) — NOT the
 * slot's `urlId` — so a second queue-capable slot gets a real,
 * independent tab with zero `reviewTabs.ts` edits and zero risk of
 * colliding with another slot's `urlId`. The `urlId` (`'plates'` for
 * `license_plate`) is kept alive purely as the bookmark contract via
 * `tabFromUrlId()` below.
 *
 * REVIEW_TABS below builds from `registeredSlots`
 * (`./annotations/registeredSlots.ts`, P2.10) — the one deployment-
 * config file listing which slots are actually live — rather than a
 * literal `[licensePlateSlot]` here, so registering a new live slot
 * there is the only edit needed to also get its review tab.
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

/** `let`, not `const` — rebuilt in place when a tier-2 deployment
 *  profile installs additional slots, before first render. See
 *  `annotations/registeredSlots.ts`'s note on live bindings. Every
 *  consumer (`{#each REVIEW_TABS}`, `tabFromUrlId`, `endpointForTab`'s
 *  default parameter) reads it at use time and needs no change. */
export let REVIEW_TABS: ReviewTabDef[] = [
  ...CORE_REVIEW_TABS,
  ...buildReviewTabs(registeredSlots),
];

onRegisteredSlotsChanged(() => {
  REVIEW_TABS = [...CORE_REVIEW_TABS, ...buildReviewTabs(registeredSlots)];
});

/** True for any tab backed by a slot's queue capability rather than a
 *  core review cohort — structural, not a registry lookup, so it stays
 *  correct even for a tab id that used to resolve to a slot that has
 *  since been unregistered. */
export function isSlotTab(id: ReviewTab): id is SlotReviewTab {
  return id.startsWith('slot:');
}

/** Resolves a `?tab=` URL value (a `QueueCapability.urlId`, or a core
 *  tab's own id) to the internal `ReviewTab`. Existing bookmarks using
 *  `?tab=plates` keep resolving to `slot:license_plate` forever, even
 *  though that string no longer appears anywhere as an internal id. */
export function tabFromUrlId(urlId: string): ReviewTab | undefined {
  const slotMatch = REVIEW_TABS.find((t) => t.slot != null && t.urlId === urlId);
  if (slotMatch) return slotMatch.id;
  const coreMatch = CORE_REVIEW_TABS.find((t) => t.urlId === urlId);
  if (coreMatch) return coreMatch.id;
  return undefined;
}

/** `{API_PREFIX}/review/{endpointId}` — what `getReviewQueue` should
 *  actually call. Falls through to the raw id for anything not in
 *  `REVIEW_TABS` (e.g. a preset id, which is already a valid endpoint
 *  segment on its own). */
export function endpointForTab(
  id: ReviewTab,
  tabs: ReviewTabDef[] = REVIEW_TABS,
): string {
  return tabs.find((t) => t.id === id)?.endpointId ?? id;
}

export type ReviewPresetId = 'mismatches' | 'gemma_low_conf' | 'primary_low_conf';

export interface ReviewPresetDef {
  id: ReviewPresetId;
  label: string;
  description: string;
}

/**
 * Quick-filter preset chips shown on the `all` tab (2026-09 tab
 * consolidation). Each reuses the exact backend cohort query the former
 * top-level tab called — `GET /curation/review/{id}` — completely unchanged;
 * only the trigger moved from a nav tab click to a chip click on top of
 * the All view. See resolveEffectiveTab for how a chip selection maps to
 * the actual queue fetched.
 */
export const REVIEW_PRESETS: ReviewPresetDef[] = [
  {
    id: 'mismatches',
    label: 'Gemma mismatches',
    description: "Gemma's suggestion disagrees with the crop's current label",
  },
  {
    id: 'gemma_low_conf',
    label: 'Gemma low-conf',
    description: "Gemma's suggestion confidence is below the review threshold",
  },
  {
    id: 'primary_low_conf',
    label: 'Primary · low-conf',
    description: 'Largest-subject-in-frame crops v6 was unsure on',
  },
];

/**
 * What `getReviewQueue` should actually be called with. Presets only ever
 * apply while the operator is on the `all` tab — every other screen
 * (Uncertainty / Model Disagreements / COCO Blind Spots / Plates) ignores
 * `preset` entirely, and navigating to one of those tabs clears it (see
 * the tab-click handler in `+page.svelte`) so a stale preset can never
 * leak into an unrelated tab's query.
 */
export function resolveEffectiveTab(
  tab: ReviewTab,
  preset: ReviewPresetId | null,
): ReviewTab {
  return tab === 'all' && preset ? preset : tab;
}
