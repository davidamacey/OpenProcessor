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
 * A review of all 9 tabs against a large production index found three
 * were too big to be curated queues — Mismatches (nearly the size of
 * "All"), VLM Low-Conf (about a tenth of the dataset), and Primary ·
 * Low-Conf (most of the ENTIRE dataset). Those three collapsed
 * into quick-filter preset chips shown on the `all` tab instead (see
 * REVIEW_PRESETS below) — same backend cohort query as before
 * (`{API_PREFIX}/review/{id}`, via resolveEffectiveTab), just triggered from a chip
 * instead of a nav tab.
 *
 * Outliers was retired entirely: 3 live rows, half its backend query
 * already dead code by its own comment, and functionally identical to the
 * `atypicality` sort already available via the strategy bar. Its backend
 * `{API_PREFIX}/review/outliers` endpoint is untouched (still reachable, just no
 * longer linked from this UI).
 *
 * Uncertainty, Model Disagreements, and Classifier Blind Spots stay as
 * top-level tabs unchanged — each is a real, distinct signal the live
 * counts backed up.
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
  // Primary-subject active-learning queue — items a proposal detector
  // found that the classifier missed entirely. A distinct failure mode.
  {
    id: 'classifier_blind_spots',
    label: 'Classifier Blind Spots',
    urlId: 'classifier_blind_spots',
    endpointId: 'classifier_blind_spots',
  },
  // New-class-proposal queue (2026-09-24 logic-moves W5) — crops the VLM
  // flagged as needing a class the registry doesn't have yet
  // (`needs_new_class`/`class_source: 'vlm_new_class_pending'`). A real
  // top-level tab, not a preset: it's a distinct triage workflow (confirm
  // vs. map-to-existing vs. create-a-class), not a filter over "All."
  // `/classes`'s Proposals section is a separate, aggregate view of the
  // same underlying cohort (GET .../new_class_proposals/summary).
  {
    id: 'new_class_proposals',
    label: 'New Class Proposals',
    urlId: 'new_class_proposals',
    endpointId: 'new_class_proposals',
  },
  // W10: items whose labels came from a labeled-dataset import. Not in the
  // tab bar unless the backend serves an `imported` entry in
  // `GET /review/tabs` (see `visibleReviewTabs`): absent, never disabled.
  {
    id: 'imported',
    label: 'Imported',
    urlId: 'imported',
    endpointId: 'imported',
  },
];

/** Tab ids shown only when the served `GET /review/tabs` vocabulary has an
 *  entry for their endpoint id. */
const SERVED_ONLY_TABS: ReadonlySet<ReviewTab> = new Set(['imported']);

/** `tabs` minus every served-only tab (`imported`) the backend does not
 *  serve. `isServed(endpointId)` answers from the served vocabulary. */
export function visibleReviewTabs(
  tabs: ReviewTabDef[],
  isServed: (endpointId: string) => boolean,
): ReviewTabDef[] {
  return tabs.filter((t) => !SERVED_ONLY_TABS.has(t.id) || isServed(t.endpointId));
}

/**
 * Slot tabs — derived from each queue-capable slot's `QueueCapability`
 * (docs/genericization-plan-2026-09-13.md §3.3/P2.8, finished by the
 * §9.5 addendum) instead of a hand-maintained literal. The internal tab
 * id is the structural `slot:${key}` template (`slotTabId`) — NOT the
 * slot's `urlId` — so a second queue-capable slot gets a real,
 * independent tab with zero `reviewTabs.ts` edits and zero risk of
 * colliding with another slot's `urlId`. The `urlId` is kept purely as
 * the bookmark contract via `tabFromUrlId()` below.
 *
 * REVIEW_TABS below builds from `registeredSlots`
 * (`./annotations/registeredSlots.ts`, P2.10) — the one deployment-
 * config file listing which slots are actually live — rather than a
 * literal slot list here, so registering a new live slot
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
 *  tab's own id) to the internal `ReviewTab`. A slot's `?tab=<urlId>`
 *  bookmark resolves to its `slot:${key}` tab even though the urlId never
 *  appears as an internal id. */
export function tabFromUrlId(urlId: string): ReviewTab | undefined {
  const slotMatch = REVIEW_TABS.find((t) => t.slot != null && t.urlId === urlId);
  if (slotMatch) return slotMatch.id;
  const coreMatch = CORE_REVIEW_TABS.find((t) => t.urlId === urlId);
  if (coreMatch) return coreMatch.id;
  return undefined;
}

/**
 * Tabs with no tuned default sort of their own — the ONLY tabs where the
 * deployment's pinned `sort` default (`GET {API_PREFIX}/settings`,
 * `SETTINGS_AXES`'s `sort` blurb in `curationSettings.ts`) actually
 * applies. Every other core tab (Uncertainty / Model Disagreements /
 * Classifier Blind Spots) and every slot tab keeps applying its own
 * `review_sorts.py`-tuned default regardless of this setting. A preset
 * chip's own endpoint id (`mismatches` / `vlm_low_conf` /
 * `primary_low_conf`) is never in this set either — each reuses its
 * former top-level tab's own tuned default, not All's absence of one —
 * so gating on `resolveEffectiveTab`'s result (not the raw nav tab)
 * already excludes them for free.
 */
export const TABS_WITH_PINNED_SORT_FALLBACK: ReadonlySet<ReviewTab> = new Set([
  'all',
  'new_class_proposals',
]);

/** True when the deployment's pinned `sort` default can apply to `tab`
 *  (see `TABS_WITH_PINNED_SORT_FALLBACK`). Pass the *effective* tab
 *  (`resolveEffectiveTab`'s result), not the raw nav tab. */
export function tabHonorsPinnedSortDefault(tab: ReviewTab): boolean {
  return TABS_WITH_PINNED_SORT_FALLBACK.has(tab);
}

/** What a `/review?tab=…&crop_id=…&preset=…` link asks for. An unknown or
 *  absent `tab` opens All; `cropId` is null when absent or empty.
 *  `preset` (m31, 2026-09-24 interactive pass) is only meaningful when
 *  `tab` resolves to `all` — validated against `REVIEW_PRESETS` here so
 *  a garbage/typo'd query value never becomes bogus selected state. */
export function reviewDeepLink(params: URLSearchParams): {
  tab: ReviewTab;
  cropId: string | null;
  preset: ReviewPresetId | null;
  /** `?import_id=` (a W10 import's own id), or null. Whether it is sent
   *  is the served `filters` list's call, not this function's. */
  importId: string | null;
  /** `?combine_conflict=true`. */
  combineConflict: boolean;
  /** The requested `?tab=` value when it resolved to no tab (so the page
   *  fell back to All) — the page says why instead of switching silently. */
  unavailableTab: string | null;
} {
  const requested = params.get('tab') ?? '';
  const resolved = tabFromUrlId(requested);
  const tab = resolved ?? 'all';
  const rawPreset = params.get('preset');
  const preset =
    tab === 'all' && rawPreset && isReviewPresetId(rawPreset) ? rawPreset : null;
  return {
    tab,
    cropId: params.get('crop_id') || null,
    preset,
    importId: params.get('import_id') || null,
    combineConflict: params.get('combine_conflict') === 'true',
    unavailableTab: requested !== '' && resolved == null ? requested : null,
  };
}

/**
 * The notice for a `?tab=` that resolved to no tab. The region tab only
 * exists while the backend serves a region profile, so a region-tab link
 * on a backend without one gets that reason, not a silent switch to All.
 */
export function unavailableTabMessage(
  requested: string,
  regionTabId: string,
  regionConfigured: boolean,
  regionUnknown = false,
): string {
  if (requested === regionTabId && regionUnknown) {
    return "The region tab isn't available yet: the region profile hasn't loaded. Showing All; it appears once the backend answers.";
  }
  if (requested === regionTabId && !regionConfigured) {
    return "The region tab isn't available: the backend reports no region profile. Showing All instead.";
  }
  return `There is no "${requested}" review tab. Showing All instead.`;
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

export type ReviewPresetId = 'mismatches' | 'vlm_low_conf' | 'primary_low_conf';

export interface ReviewPresetDef {
  id: ReviewPresetId;
  label: string;
  description: string;
}

/**
 * Quick-filter preset chips shown on the `all` tab (2026-09 tab
 * consolidation). Each reuses the exact backend cohort query the former
 * top-level tab called — `GET {API_PREFIX}/review/{id}` — completely unchanged;
 * only the trigger moved from a nav tab click to a chip click on top of
 * the All view. See resolveEffectiveTab for how a chip selection maps to
 * the actual queue fetched.
 */
export const REVIEW_PRESETS: ReviewPresetDef[] = [
  {
    id: 'mismatches',
    label: 'VLM mismatches',
    description: "The VLM's suggestion disagrees with the crop's current label",
  },
  {
    id: 'vlm_low_conf',
    label: 'VLM low-conf',
    description: "The VLM's suggestion confidence is below the review threshold",
  },
  {
    id: 'primary_low_conf',
    label: 'Primary · low-conf',
    description: 'Largest-subject-in-frame crops v6 was unsure on',
  },
];

/** Type guard for a `?preset=` query value — function declaration (not a
 *  const) so it's usable from `reviewDeepLink` above regardless of
 *  declaration order in this module. */
export function isReviewPresetId(value: string): value is ReviewPresetId {
  return REVIEW_PRESETS.some((p) => p.id === value);
}

/**
 * What `getReviewQueue` should actually be called with. Presets only ever
 * apply while the operator is on the `all` tab — every other screen
 * (Uncertainty / Model Disagreements / Classifier Blind Spots / slot tabs) ignores
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
