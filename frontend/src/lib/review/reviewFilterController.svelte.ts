/**
 * `/review` filter bar state: the Source box, the shared item filter, the
 * confidence range, the slot text search, the served per-tab enum filters,
 * the URL-seeded `import_id`/`combine_conflict` chips, the subject scope and
 * clarity slider, and `_filter()`, the one place they become query params.
 *
 * Moved verbatim out of `review/+page.svelte`.
 */

import { resolve } from '$app/paths';
import { page } from '$app/state';
import { replaceState } from '$app/navigation';
import { projectHref } from '$lib/projectPaths';
import {
  ItemFilterState,
  withoutOpenVocab,
} from '$lib/itemFilter/itemFilterState.svelte';
import { ITEM_FILTER_PARAMS } from '$lib/itemFilter/itemFilterControls';
import type { ReviewDiverseController } from '$lib/review/reviewDiverseController.svelte';
import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';

export type ServedFilterValue = string | string[];

export interface ReviewFilterOptions {
  tabs: ReviewTabsController;
  diverse: ReviewDiverseController;
}

export function createReviewFilterController(opts: ReviewFilterOptions) {
  const { tabs, diverse } = opts;
  const { deepLink, filterVisible } = tabs;
  const strategyBar = diverse.strategyBar;
  const activeSlot = $derived(tabs.activeSlot);
  const activeTabEndpointId = $derived(tabs.activeTabEndpointId);
  const diverseMode = $derived(diverse.diverseMode);

  // Filter bar. `sourceFilter` sends `source` to {API_PREFIX}/review/{tab};
  // `termFilters()` below (the diverse-selection scope, a different
  // endpoint) sends the same value under `source` too.
  let sourceFilter = $state<string>('');
  // The shared item filter (class by name, origin, embedding / review state,
  // area band), seeded from the URL and persisted back to it. Conf and the
  // largest-N toggle keep their own controls below; the bar never draws them.
  const itemFilter = new ItemFilterState();
  itemFilter.fromUrl(page.url.searchParams);
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- constant lookup set, built once and never mutated
  const PAGE_OWNED_PARAMS = new Set(['conf_min', 'conf_max', 'max_rank']);
  function itemFilterVisible(param: string): boolean {
    if (PAGE_OWNED_PARAMS.has(param) || !withoutOpenVocab(param)) return false;
    // POST /select/diverse's scope.filters only takes exact terms.
    if (diverseMode) return param === 'class_name';
    return filterVisible(param);
  }
  const itemFilterQuery = $derived(
    itemFilter.toQuery(
      (p) => !PAGE_OWNED_PARAMS.has(p) && withoutOpenVocab(p) && filterVisible(p),
    ),
  );
  function persistItemFilter(): void {
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local URL built and handed to replaceState synchronously, never stored in reactive state
    const url = new URL(page.url.href);
    itemFilter.toUrl(url.searchParams);
    replaceState(resolve(projectHref(`/review${url.search}`)), {});
  }
  let confMin = $state<number>(0);
  let confMax = $state<number>(1);
  // Slot text search — only meaningful on a slot tab with a text filter;
  // ignored elsewhere server-side.
  let slotTextQuery = $state<string>('');

  // Generic served-enum filter bar (3f1a11e adoption) — one entry per
  // `ReviewFilterSpec.param` the active tab declares (e.g. `region_status`
  // on the Regions tab). No param-specific code here or in `_filter()`
  // below: a future spec on any tab just works. Reset whenever the tab
  // changes (see `resetForTab` below); seeded from the URL on
  // first load so `?region_status=verify_rejected` is bookmarkable, the
  // same pattern `preset` uses.
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- constant lookup set, built once and never mutated
  const NON_ENUM_FILTER_PARAMS = new Set([
    'tab',
    'preset',
    'crop_id',
    'import_id',
    'combine_conflict',
    ...ITEM_FILTER_PARAMS,
  ]);
  let enumFilterValues = $state<Record<string, ServedFilterValue>>(
    Object.fromEntries(
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local dedup of the URL keys, consumed synchronously and never stored
      [...new Set(page.url.searchParams.keys())]
        .filter((k) => !NON_ENUM_FILTER_PARAMS.has(k))
        .map((k): [string, ServedFilterValue] => {
          const all = page.url.searchParams.getAll(k);
          return [k, all.length > 1 ? all : all[0]!];
        }),
    ),
  );
  const servedSpecs = $derived(
    reviewTabsVocabularyStore.filterSpecsFor(activeTabEndpointId),
  );
  // Params this page draws itself (the shared bar, the Source box, the
  // strategy bar's sort / mistakenness / near-duplicate controls, the blur
  // slider, the slot text box and the URL-seeded chips); every other served
  // spec is drawn generically by its kind.
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- constant lookup set, built once and never mutated
  const SELF_DRAWN_PARAMS = new Set([
    ...ITEM_FILTER_PARAMS,
    'source',
    'sort',
    'min_blur_ratio',
    'min_mistakenness',
    'hide_near_duplicates',
    'import_id',
    'combine_conflict',
  ]);
  const activeFilterSpecs = $derived(
    servedSpecs.filter(
      (s) =>
        !SELF_DRAWN_PARAMS.has(s.param) &&
        s.param !== activeSlot?.capabilities.queue?.textFilter?.param,
    ),
  );
  // Exactly the enum params _filter() sends: only those the active tab's
  // served filter_specs declare. The refetch effect keys on this, not on
  // enumFilterValues, so a URL-seeded ?region_status= that arrives before
  // /review/tabs has loaded still triggers a refetch once the spec lands.
  const activeEnumParams = $derived.by<Record<string, ServedFilterValue>>(() => {
    const out: Record<string, ServedFilterValue> = {};
    for (const spec of activeFilterSpecs) {
      const value = enumFilterValues[spec.param];
      if (value && value.length > 0) out[spec.param] = value;
    }
    return out;
  });
  // URL-seeded non-enum filters (`import_id`, `combine_conflict`):
  // no toggle of their own (they would be noise in every project), only a
  // removable chip when a link carries them. Sent to `/review/{tab}` and
  // `/locate` only on a tab whose served `filters` list them: unlike the
  // filter bar's own controls (which stay visible while the vocabulary is
  // unknown), an unlisted URL param is never guessed, so nothing is sent
  // before `GET /review/tabs` has answered.
  let importIdFilter = $state<string>(deepLink.importId ?? '');
  let combineConflictFilter = $state<boolean>(deepLink.combineConflict);
  function urlFilterServed(param: string): boolean {
    return (
      reviewTabsVocabularyStore.filtersFor(activeTabEndpointId)?.includes(param) === true
    );
  }
  // A link that seeds one of these holds the first queue load until the
  // vocabulary has answered (it settles even on failure), so the queue never
  // flashes unfiltered first.
  const waitingForVocabulary = $derived(
    (importIdFilter !== '' || combineConflictFilter) && !reviewTabsVocabularyStore.loaded,
  );
  const activeUrlFilters = $derived.by<Record<string, string | boolean>>(() => {
    const out: Record<string, string | boolean> = {};
    if (importIdFilter && urlFilterServed('import_id')) out.import_id = importIdFilter;
    if (combineConflictFilter && urlFilterServed('combine_conflict')) {
      out.combine_conflict = true;
    }
    return out;
  });
  function clearUrlFilter(param: 'import_id' | 'combine_conflict'): void {
    if (param === 'import_id') importIdFilter = '';
    else combineConflictFilter = false;
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local URL built and handed to replaceState synchronously, never stored in reactive state
    const url = new URL(page.url.href);
    url.searchParams.delete(param);
    replaceState(resolve(projectHref(`/review${url.search}`)), {});
  }
  function setEnumFilter(param: string, value: ServedFilterValue): void {
    enumFilterValues = { ...enumFilterValues, [param]: value };
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local URL built and handed to replaceState synchronously, never stored in reactive state
    const url = new URL(page.url.href);
    url.searchParams.delete(param);
    for (const v of Array.isArray(value) ? value : value ? [value] : []) {
      url.searchParams.append(param, v);
    }
    replaceState(resolve(projectHref(`/review${url.search}`)), {});
  }

  /** The per-tab filters a tab switch drops, in state and in `url`. */
  function resetForTab(url: URL): void {
    // Generic served-enum filters are per-tab — a param from the
    // previous tab (e.g. Regions' `region_status`) never carries
    // into the new one, in state or the URL.
    for (const param of Object.keys(enumFilterValues)) {
      url.searchParams.delete(param);
    }
    enumFilterValues = {};
    itemFilter.clear();
    itemFilter.toUrl(url.searchParams);
    // The URL-seeded filters are per-tab too.
    url.searchParams.delete('import_id');
    url.searchParams.delete('combine_conflict');
    importIdFilter = '';
    combineConflictFilter = false;
  }

  // Primary-subject controls (primary_low_conf / classifier_blind_spots tabs).
  // subjectScope: 1 = largest only, 2 = largest + 2nd (the tabs default to 2
  // server-side when unset). Clarity slider commits on release.
  let subjectScope = $state<0 | 1 | 2>(0);
  const BLUR_MAX = 2;
  let blurSlider = $state<number>(0);
  let minBlurRatio = $state<number | null>(null);
  function commitBlur(): void {
    minBlurRatio = blurSlider > 0 ? blurSlider : null;
  }

  function _filter(): Record<string, unknown> {
    const f: Record<string, unknown> = {};
    // GET /review/{tab} accepts class_name (by name, repeatable)/source/conf_min/conf_max as of
    // the 2026-09-24 logic-moves cutover (item 14/G3 — verified live
    // against the real backend). Diverse mode disables these controls
    // (see the filter bar below) since POST {API_PREFIX}/select/diverse's
    // `scope.filters` doesn't support conf_min/conf_max at all, and
    // takes class_name/source through its own `termFilters()` instead of
    // this function.
    Object.assign(f, itemFilterQuery);
    if (sourceFilter) f.source = sourceFilter;
    if (confMin > 0) f.conf_min = confMin;
    if (confMax < 1) f.conf_max = confMax;
    const textFilter = activeSlot?.capabilities.queue?.textFilter;
    if (textFilter && slotTextQuery) f[textFilter.param] = slotTextQuery;
    // max_rank / min_blur_ratio apply across every tab and preset — the
    // backend's own review.py comment says so explicitly ("Both apply
    // across tabs"). These used to be gated to only primary_low_conf /
    // classifier_blind_spots, which meant the rank-scope and clarity controls
    // silently appeared/disappeared depending on which tab or quick-filter
    // chip was active — confusing and inconsistent with Conf/Class/Source,
    // which were never gated. Always available now, like those.
    if (subjectScope !== 0) f.max_rank = subjectScope;
    if (minBlurRatio != null) f.min_blur_ratio = minBlurRatio;
    // Generic served-enum filters (3f1a11e adoption) — sent whenever the
    // operator picked a value; the backend applies its own
    // `filter_defaults` when a param is omitted, so an unset control
    // never needs a client-side default to fall back to. Only params the
    // active tab's served filter_specs declare are forwarded: the state is
    // seeded from every URL param, so an unrelated or stale one (e.g.
    // ?region_status= on a tab without that spec) must not reach the
    // backend, which 400s on filters a tab doesn't honor.
    Object.assign(f, activeEnumParams);
    Object.assign(f, activeUrlFilters);
    Object.assign(f, strategyBar.toQueryParams());
    return f;
  }

  // R3: the operator's own narrowing filters (never the strategy bar's
  // sort) — an empty queue under these may just be filtered empty.
  const clientFiltersActive = $derived(
    !itemFilter.isEmpty ||
      sourceFilter !== '' ||
      confMin > 0 ||
      confMax < 1 ||
      slotTextQuery !== '' ||
      subjectScope !== 0 ||
      minBlurRatio != null ||
      Object.keys(activeEnumParams).length > 0 ||
      Object.keys(activeUrlFilters).length > 0,
  );

  return {
    get sourceFilter() {
      return sourceFilter;
    },
    set sourceFilter(v: string) {
      sourceFilter = v;
    },
    itemFilter,
    itemFilterVisible,
    get itemFilterQuery() {
      return itemFilterQuery;
    },
    persistItemFilter,
    get confMin() {
      return confMin;
    },
    set confMin(v: number) {
      confMin = v;
    },
    get confMax() {
      return confMax;
    },
    set confMax(v: number) {
      confMax = v;
    },
    get slotTextQuery() {
      return slotTextQuery;
    },
    set slotTextQuery(v: string) {
      slotTextQuery = v;
    },
    get enumFilterValues() {
      return enumFilterValues;
    },
    get servedSpecs() {
      return servedSpecs;
    },
    get activeFilterSpecs() {
      return activeFilterSpecs;
    },
    get activeEnumParams() {
      return activeEnumParams;
    },
    get importIdFilter() {
      return importIdFilter;
    },
    get combineConflictFilter() {
      return combineConflictFilter;
    },
    urlFilterServed,
    get waitingForVocabulary() {
      return waitingForVocabulary;
    },
    get activeUrlFilters() {
      return activeUrlFilters;
    },
    clearUrlFilter,
    setEnumFilter,
    resetForTab,
    get subjectScope() {
      return subjectScope;
    },
    set subjectScope(v: 0 | 1 | 2) {
      subjectScope = v;
    },
    BLUR_MAX,
    get blurSlider() {
      return blurSlider;
    },
    set blurSlider(v: number) {
      blurSlider = v;
    },
    get minBlurRatio() {
      return minBlurRatio;
    },
    commitBlur,
    _filter,
    get clientFiltersActive() {
      return clientFiltersActive;
    },
  };
}

export type ReviewFilterController = ReturnType<typeof createReviewFilterController>;
