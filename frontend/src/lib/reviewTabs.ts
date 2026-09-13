import type { ReviewTab } from './types';

export interface ReviewTabDef {
  id: ReviewTab;
  label: string;
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
 * Uncertainty, Model Disagreements, COCO Blind Spots, and Plates stay as
 * top-level tabs unchanged — each is a real, distinct signal the live
 * counts backed up (100 / 12 / 1,000 / 76 respectively).
 */
export const REVIEW_TABS: ReviewTabDef[] = [
  { id: 'all', label: 'All' },
  { id: 'uncertainty', label: 'Uncertainty' },
  // Phase 5 active-learning loop: validated crops where the newly
  // promoted model disagrees with the human label.
  { id: 'model_disagreements', label: 'Model Disagreements' },
  // Primary-subject active-learning queue — COCO-confirmed vehicles v6
  // missed entirely. Genuinely distinct failure mode from the rest.
  { id: 'coco_blind_spots', label: 'COCO Blind Spots' },
  // Plate-detection review: crops with an LPR/SAM3+Gemma-verified plate
  // bbox waiting for human confirmation in SlotBboxEditor. Entirely separate
  // workflow/object type from vehicle-class review.
  { id: 'plates', label: 'Plates' },
];

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
