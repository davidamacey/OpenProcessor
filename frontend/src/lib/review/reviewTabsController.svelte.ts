/**
 * `/review` tab state: the active tab and quick-filter preset, the
 * `?tab=`/`?crop_id=`/`?preset=` deep link they are seeded from, and the
 * served per-tab vocabulary reads (visible tabs, filter visibility, the
 * served `max_rank` default, the active queue's label).
 *
 * Moved verbatim out of `review/+page.svelte`. The tab switch resets the
 * per-tab filters and closes the class picker through the two callbacks the
 * page hands in, so this module never imports the filter or picker
 * controllers.
 */

import { resolve } from '$app/paths';
import { page } from '$app/state';
import { replaceState } from '$app/navigation';
import { projectHref } from '$lib/projectPaths';
import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
import {
  endpointForTab,
  isServedOnlyTab,
  REVIEW_PRESETS,
  REVIEW_TABS,
  resolveEffectiveTab,
  visibleReviewTabs,
  type ReviewPresetId,
  type ReviewTabDef,
  reviewDeepLink,
  unavailableTabMessage,
} from '$lib/reviewTabs';
import { REGION_TAB_ID } from '$lib/annotations/servedRegionSlot';
import { regionProfileStore } from '$stores/regionProfile.svelte';
import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';
import type { ReviewTab } from '$lib/types_review';

export interface ReviewTabsOptions {
  /** Drop the previous tab's filters, in state and in `url`. */
  resetTabFilters: (url: URL) => void;
  closePicker: () => void;
}

export function createReviewTabsController(opts: ReviewTabsOptions) {
  // Unified review by default — one continuous queue of every crop that
  // needs a human, sorted most-uncertain first. Down to 5 top-level tabs
  // (2026-09 consolidation, see $lib/reviewTabs.ts) — Mismatches / VLM
  // Low-Conf / Primary·Low-Conf collapsed into preset chips on the All
  // tab (below); Outliers retired entirely (see reviewTabs.ts doc
  // comment). The remaining 4 narrower tabs stay available for
  // diagnosing where uncertainty came from — each is a real, distinct
  // signal, not a rebrand of "everything." `new_class_proposals` is a 6th real tab, not a preset — a
  // distinct triage workflow (confirm / map-to-existing / create-a-class)
  // over crops the VLM flagged as needing a class the registry doesn't
  // have yet.
  // `/review?tab=<urlId>&crop_id=<id>` deep links (bookmarks, /train's
  // cohort preview) open that tab and jump to that crop.
  const deepLink = reviewDeepLink(page.url.searchParams);
  let tab = $state<ReviewTab>(deepLink.tab);
  // A `?tab=` that resolved to no tab (e.g. the region tab on a backend
  // with no region profile) falls back to All with a visible reason.
  let unavailableTabNotice = $state<string | null>(
    deepLink.unavailableTab
      ? unavailableTabMessage(
          deepLink.unavailableTab,
          REGION_TAB_ID,
          regionProfileStore.configured,
          regionProfileStore.unknown,
        )
      : null,
  );
  let pendingCropId = $state<string | null>(deepLink.cropId);
  // DQ-M7 (2026-09-24 data-quality pass): true from mount until a
  // `?crop_id=` deep link either lands on its target or gives up. While
  // true, the tab-action keybinding effect below registers nothing, so a
  // keypress during the ~4s a deep link to a late page used to spend
  // showing item #1 (while paging 1..page sequentially) can no longer act
  // on the wrong crop. False immediately when there's no deep link to
  // resolve.
  let awaitingDeepLink = $state<boolean>(deepLink.cropId != null);
  // The slot backing the current tab, if any — the single derived value
  // P2.8b's mapping table (docs/genericization-plan-2026-09-13.md §9.5)
  // hangs every slot-tab call site off, instead of a hand-maintained
  // literal per site.
  const activeSlot = $derived(REVIEW_TABS.find((t) => t.id === tab)?.slot ?? null);
  // The `imported` tab is offered only while `GET /review/tabs` serves it.
  const visibleTabs = $derived(
    visibleReviewTabs(REVIEW_TABS, (id) => reviewTabsVocabularyStore.hasEntry(id)),
  );
  // The imported tab's empty state links to the import page only when
  // dataset import is served; probe lazily, never on tabs that don't need it.
  $effect(() => {
    if (tab === 'imported') void datasetsAvailability.init();
  });
  // A `?tab=` link to a served-only tab (`imported`,
  // `detector_disagreements`) against a backend whose vocabulary lacks it
  // falls back to All once the vocabulary has loaded.
  $effect(() => {
    if (!reviewTabsVocabularyStore.loaded || !isServedOnlyTab(tab)) return;
    if (reviewTabsVocabularyStore.hasEntry(endpointForTab(tab))) return;
    const missing = endpointForTab(tab);
    tab = 'all';
    unavailableTabNotice = unavailableTabMessage(
      missing,
      REGION_TAB_ID,
      regionProfileStore.configured,
      regionProfileStore.unknown,
    );
  });
  // Active quick-filter preset chip on the All tab (null = plain All).
  // Only ever meaningful while tab === 'all' — resolveEffectiveTab drops
  // it for every other tab, and switching tabs clears it outright.
  // m31 (2026-09-24 interactive pass): seeded from `?preset=` so a
  // preset chip is bookmarkable — it used to reset to plain All on
  // every reload/share, silently dropping the filter.
  let preset = $state<ReviewPresetId | null>(deepLink.preset);
  const effectiveTab = $derived<ReviewTab>(resolveEffectiveTab(tab, preset));
  // dq-queues cutover (2026-09-24): GET {API_PREFIX}/review/tabs now serves each
  // tab's `filters`/`filter_defaults` — the filter bar renders only the
  // controls the active tab's served entry lists, and the subject/
  // max_rank control's "unset" label reflects the served default
  // instead of a hardcoded "Top 2". `null` (endpoint absent/older
  // backend) means "unknown" and every control still renders, same as
  // before this landed.
  const activeTabEndpointId = $derived(endpointForTab(effectiveTab));
  function filterVisible(param: string): boolean {
    return reviewTabsVocabularyStore.filterSupported(activeTabEndpointId, param);
  }
  const servedMaxRankDefault = $derived.by<number | null>(() => {
    const v = reviewTabsVocabularyStore.filterDefault(activeTabEndpointId, 'max_rank');
    return typeof v === 'number' ? v : null;
  });
  function togglePreset(id: ReviewPresetId): void {
    preset = preset === id ? null : id;
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local URL built and handed to replaceState synchronously, never stored in reactive state
    const url = new URL(page.url.href);
    if (preset) {
      url.searchParams.set('preset', preset);
    } else {
      url.searchParams.delete('preset');
    }
    replaceState(resolve(projectHref(`/review${url.search}`)), {});
  }

  const activeQueueLabel = $derived(
    preset
      ? reviewTabsVocabularyStore.labelFor(
          preset,
          REVIEW_PRESETS.find((p) => p.id === preset)?.label ?? preset,
        )
      : reviewTabsVocabularyStore.labelFor(
          activeTabEndpointId,
          REVIEW_TABS.find((t) => t.id === tab)?.label ?? String(tab),
        ),
  );

  function selectTab(t: ReviewTabDef): void {
    tab = t.id;
    pendingCropId = null;
    awaitingDeepLink = false;
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local URL built and handed to replaceState synchronously, never stored in reactive state
    const url = new URL(page.url.href);
    url.searchParams.set('tab', t.urlId);
    url.searchParams.delete('crop_id');
    url.searchParams.delete('preset');
    // Generic served-enum filters, the shared item filter and the
    // URL-seeded filters are per-tab: none carries into the new tab.
    opts.resetTabFilters(url);
    replaceState(resolve(projectHref(`/review${url.search}`)), {});
    // Presets only make sense on the All tab — switching to any
    // other tab (or re-landing on All from one) always starts
    // from plain All rather than silently carrying a stale chip.
    preset = null;
    opts.closePicker();
  }

  return {
    deepLink,
    get tab() {
      return tab;
    },
    get unavailableTabNotice() {
      return unavailableTabNotice;
    },
    set unavailableTabNotice(v: string | null) {
      unavailableTabNotice = v;
    },
    get pendingCropId() {
      return pendingCropId;
    },
    set pendingCropId(v: string | null) {
      pendingCropId = v;
    },
    get awaitingDeepLink() {
      return awaitingDeepLink;
    },
    set awaitingDeepLink(v: boolean) {
      awaitingDeepLink = v;
    },
    get activeSlot() {
      return activeSlot;
    },
    get visibleTabs() {
      return visibleTabs;
    },
    get preset() {
      return preset;
    },
    get effectiveTab() {
      return effectiveTab;
    },
    get activeTabEndpointId() {
      return activeTabEndpointId;
    },
    filterVisible,
    get servedMaxRankDefault() {
      return servedMaxRankDefault;
    },
    togglePreset,
    get activeQueueLabel() {
      return activeQueueLabel;
    },
    selectTab,
  };
}

export type ReviewTabsController = ReturnType<typeof createReviewTabsController>;
