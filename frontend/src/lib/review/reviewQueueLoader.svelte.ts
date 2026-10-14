/**
 * `/review` queue: the pager, the cursor, the handled-ids dedup set, the
 * `?crop_id=` locate jump, the semantic-search result swap, and the ordered
 * refetch effects (locate, immediate,
 * debounced, empty-tab) that load it.
 *
 * Moved verbatim out of `review/+page.svelte`. `queue`, `cursor` and
 * `handledIds` live here, once, and every other review controller reads and
 * writes them through this object (by reference), never through a copy. The
 * refetch effects stay together in this one module because their order is
 * load-bearing: the immediate effect loads page 1, the debounced one only
 * records the starting filters on its first run, and the locate effect
 * waits for that first page.
 */

import { untrack } from 'svelte';
import { apiErrorText } from '$lib/api_errors';
import { getReviewQueue, locateInReviewQueue } from '$lib/api_review';
import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
import { createPager } from '$lib/pager.svelte';
import { createReviewQueueController } from '$lib/review/reviewController.svelte';
import type { ReviewDiverseController } from '$lib/review/reviewDiverseController.svelte';
import type { ReviewFilterController } from '$lib/review/reviewFilterController.svelte';
import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
import {
  emptyQueueMessage,
  locateMissMessage,
  queuePosition,
} from '$lib/review/reviewCopy';
import { endpointForTab } from '$lib/reviewTabs';
import type { SemanticSearchResult } from '$lib/searchBox.svelte';
import { isSemanticSearchAvailable } from '$lib/strategies';
import type { ReviewItem, ReviewTab } from '$lib/types_review';
import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';
import { strategiesStore } from '$stores/strategies.svelte';
import { toastStore } from '$stores/toast.svelte';

/** Served page size of the review queue (and of the diverse overlay's pages). */
export const REVIEW_PAGE_SIZE = 30;

export interface ReviewQueueLoaderOptions {
  tabs: ReviewTabsController;
  diverse: ReviewDiverseController;
  filters: ReviewFilterController;
}

export function createReviewQueueLoader(opts: ReviewQueueLoaderOptions) {
  const { tabs, diverse, filters } = opts;
  const { _filter } = filters;
  const strategyBar = diverse.strategyBar;
  const tab = $derived(tabs.tab);
  const preset = $derived(tabs.preset);
  const effectiveTab = $derived(tabs.effectiveTab);
  const activeTabEndpointId = $derived(tabs.activeTabEndpointId);
  const diverseMode = $derived(diverse.diverseMode);
  const waitingForVocabulary = $derived(filters.waitingForVocabulary);
  const clientFiltersActive = $derived(filters.clientFiltersActive);
  const sourceFilter = $derived(filters.sourceFilter);
  const slotTextQuery = $derived(filters.slotTextQuery);
  const confMin = $derived(filters.confMin);
  const confMax = $derived(filters.confMax);
  const subjectScope = $derived(filters.subjectScope);
  const minBlurRatio = $derived(filters.minBlurRatio);
  const activeEnumParams = $derived(filters.activeEnumParams);
  const activeUrlFilters = $derived(filters.activeUrlFilters);
  const itemFilterQuery = $derived(filters.itemFilterQuery);
  const cancelDiverseJob = () => diverse.cancelDiverseJob();

  const pageSize = REVIEW_PAGE_SIZE;
  let cursor = $state<number>(0); // index within accumulated items

  // Set from the review-queue response whenever the requested `?sort=`
  // couldn't be honored server-side (e.g. the field isn't backfilled
  // yet). Rendered as a small inline note, never a toast — this isn't a
  // failure, just a degraded request.
  let sortFallbackReason = $state<string | null>(null);
  // #36 item 9: the server's own explanation for an empty queue (e.g. "no
  // probe predictions — run a probe"), distinct from sortFallbackReason.
  let emptyReason = $state<string | null>(null);
  // The sort id the backend actually applied (item 10, 2026-09-24
  // logic-moves) — passed to StrategyBar so its summary chip can show
  // it next to whatever the operator picked (or didn't).
  let sortApplied = $state<string | null>(null);

  // Queue pager. One fetchPage closure means the tab + filter set can't
  // drift between page 1 and the pages the cursor pulls in behind it.
  const queue = createPager<ReviewItem>({
    fetchPage: async (page) => {
      if (diverseMode) return diverse.fetchDiversePage(page);
      // effectiveTab is an internal id (slot:${key} for a slot tab,
      // per P2.8b) — the backend still expects the endpointId
      // (e.g. 'regions'), so this resolves through endpointForTab()
      // rather than forwarding the internal id directly.
      const res = await getReviewQueue(
        endpointForTab(effectiveTab) as ReviewTab,
        page,
        pageSize,
        _filter(),
      );
      sortFallbackReason = res.sort_fallback_reason ?? null;
      sortApplied = res.sort_applied ?? null;
      emptyReason = res.empty_reason ?? null;
      return res;
    },
    keyOf: (i) => i.id,
    onReset: () => {
      cursor = 0;
      handledIds.clear();
    },
    accept: (i) => !handledIds.has(i.id),
  });

  // Eagerly prefetch the next page when the cursor is within this many
  // items of the end of the loaded buffer. Without this, the user sees
  // 'no more' for one render-frame whenever they confirm the last item
  // we've loaded — refreshing the page then shows there were more all
  // along.
  const PREFETCH_AHEAD = 5;
  function maybePrefetch(): void {
    if (searchModeActive) return;
    if (queue.loadingMore || !queue.hasMore) return;
    if (queue.items.length - cursor <= PREFETCH_AHEAD) {
      void loadMore();
    }
  }

  // P2-14 semantic text search. Gated behind isSemanticSearchAvailable
  // exactly like every other overlay control — an old/flag-off backend
  // hides <SemanticSearchBox> entirely. Results feed straight into
  // `queue`'s settable items/total (searchScores keyed by crop id, for
  // the similarity badge alongside the existing mistakenness ScoreChip
  // below) so the existing one-at-a-time review UI, label hotkeys, and
  // undo/discard flows keep working completely unchanged. While a
  // search is active, prefetch/loadMore is disabled (see maybePrefetch
  // above) — {API_PREFIX}/search/text pagination isn't wired to this page's
  // page-N `queue.loadMore`, and paging into getReviewQueue while search
  // results are showing would silently overwrite them.
  const semanticSearchAvailable = $derived(
    isSemanticSearchAvailable(strategiesStore.methods.overlays),
  );
  let searchModeActive = $state(false);
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced wholesale on every search, never mutated in place
  let searchScores = $state(new Map<string, number>());

  function applySearchResults(res: SemanticSearchResult): void {
    searchModeActive = true;
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced wholesale on every search, never mutated in place
    searchScores = new Map(res.items.map((it) => [it.id, it.similarity_score]));
    cursor = 0;
    handledIds.clear();
    queue.items = res.items.map((it) => {
      // it already carries its own served proposed_class_id/_name
      // (item 11, 2026-09-24 logic-moves — searchCrops maps
      // through the same mapRawCrop as getReviewQueue) — no
      // client fill-in.
      const { similarity_score: _score, ...rest } = it;
      return { ...rest, reason: '' };
    });
    queue.total = res.total;
  }

  function clearSearch(): void {
    searchModeActive = false;
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- replaced wholesale on every search, never mutated in place
    searchScores = new Map();
    void queue.loadFirst();
  }

  // Crop ids this session has already assigned / discarded / triaged.
  // The queue is page-numbered over a server collection that SHRINKS as
  // you label, so page N+1 can contain a crop that page N would have held
  // before the shift. loadMore's dedup only compares against the items
  // still in the buffer — a handled crop was removed from that buffer, so
  // it would sail straight back into the queue. Entries come back out on
  // rollback and on undo-restore. Not $state: only loadMore reads it, and
  // that read is inside an async callback, never in a reactive context.
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- deliberately not reactive, see the comment above: only read inside an async callback, never in a reactive context
  const handledIds = new Set<string>();

  // Queue action controller (assign / discard / skip / undo) — extracted
  // to src/lib/review/reviewController.svelte.ts (P1-4,
  // docs/design/test-audit-2026-09-24.md) so the optimistic-remove +
  // rollback-on-failure logic is unit-testable off the page. `queue`,
  // `cursor` and `handledIds` stay owned here (shared with slot-tab
  // actions and arrow-key nav) and are handed in by
  // reference/accessor.
  const queueController = createReviewQueueController({
    queue,
    handledIds,
    getCursor: () => cursor,
    setCursor: (v) => {
      cursor = v;
    },
    maybePrefetch: () => maybePrefetch(),
  });

  // Review is one-at-a-time — only `current` is rendered. We fetch one
  // page on tab/filter change and the cursor-arrow handler pulls the
  // next page as the user nears the end. The earlier eager prefetch
  // drained every page upfront, which on the busy 'all' tab fired ~4
  // chained network calls before first paint and made the page feel
  // frozen on slow connections. Lazy paging keeps first-paint snappy.
  // F8 D6: the served queue position, not the index within the loaded
  // buffer (a deep link loads only the located page).
  const currentPosition = $derived(queuePosition(queue.firstPage, pageSize, cursor));

  const loadFirst = () => queue.loadFirst();
  const loadMore = () => queue.loadMore();

  // `/review?crop_id=` deep link:
  // ask the backend exactly where the crop sits under the active tab's
  // filters/sort via GET {API_PREFIX}/review/{tab}/locate, rather than the old
  // approach of blindly paging forward up to 300 items hoping to find
  // it. `in_queue: false` means it doesn't match this tab (already
  // handled, filtered out, etc.) — `reason` explains why when the
  // backend sends one.
  let jumpingToCrop = $state(false);
  async function jumpToPendingCrop(): Promise<void> {
    const cropId = tabs.pendingCropId;
    if (cropId == null) return;
    if (diverseMode || searchModeActive) {
      // Neither a pool-scale overlay selection nor a semantic-search
      // result set has a stable server-side "locate" — drop the deep
      // link rather than spin forever waiting for a match that can
      // never resolve.
      tabs.pendingCropId = null;
      tabs.awaitingDeepLink = false;
      return;
    }
    jumpingToCrop = true;
    try {
      const loc = await locateInReviewQueue(
        endpointForTab(effectiveTab),
        cropId,
        pageSize,
        _filter(),
      );
      if (!loc.in_queue || loc.page == null) {
        toastStore.info(locateMissMessage(loc.reason));
        return;
      }
      // M11: /locate resolves the sort under the same rules {API_PREFIX}/review/{tab}
      // does, and can hit the same 0%-coverage fallback — surface it the
      // same way so a deep link doesn't silently jump using a different
      // sort than what the bar shows.
      sortApplied = loc.sort_applied ?? sortApplied;
      sortFallbackReason = loc.sort_fallback_reason ?? sortFallbackReason;
      // DQ-M7: fetch ONLY the located page — no more paging 1..loc.page
      // one request per page (103 requests / 5.7s at rank 3000, page 101).
      // The initial page-1 load that seeded `queue` (from the tab/filter
      // effect) gets replaced wholesale here rather than ever being shown.
      await queue.loadPage(loc.page);
      const idx = queue.items.findIndex((i) => i.id === cropId);
      cursor =
        idx >= 0 ? idx : Math.max(0, Math.min(loc.rank ?? 0, queue.items.length - 1));
    } catch (e) {
      toastStore.error(`Locate failed: ${apiErrorText(e)}`);
    } finally {
      jumpingToCrop = false;
      tabs.pendingCropId = null;
      tabs.awaitingDeepLink = false;
    }
  }

  $effect(() => {
    if (
      tabs.pendingCropId == null ||
      queue.loading ||
      queue.loadingMore ||
      jumpingToCrop
    ) {
      return;
    }
    if (queue.loadedPages === 0) return; // first page not in yet
    void jumpToPendingCrop();
  });

  // Tab + class filter fire loadFirst() immediately (single-click changes
  // are intentional). Text + slider filters debounce by 250ms so typing
  // sourceFilter or dragging the confidence sliders doesn't cause a refetch
  // per keystroke.
  // Guards this effect the same way lastFilterKey guards the debounced one
  // below: observed live, this effect's body can execute an extra time
  // for the same tab/preset/subjectScope/minBlurRatio values
  // (a harmless Svelte/SvelteKit-dev re-run, not a real dependency
  // change) — without a same-key guard, that spurious extra run still
  // unconditionally cancels+invalidates the in-flight diverse selection
  // (untrack() only stops it from *looping*, not from firing once extra),
  // which can land right after a real selectDiverse POST and cancel a job
  // the operator never asked to cancel.
  let lastImmediateKey: string | null = null;
  $effect(() => {
    void tab;
    void preset;
    void subjectScope;
    void minBlurRatio;
    if (waitingForVocabulary) return;
    const key = JSON.stringify([tab, preset, subjectScope, minBlurRatio]);
    if (key === lastImmediateKey) return;
    lastImmediateKey = key;
    // effectiveTab/class-name changes invalidate any in-progress diverse
    // selection (P2-10) — the pool it was drawn from no longer matches
    // the current scope. Cancel any running job too; a stale poll left
    // running after the operator moved on would eventually resolve into
    // diverseSelection for the wrong tab. untrack() is load-bearing here:
    // cancelDiverseJob() reads diverseJobId, and runDiverseSelection()
    // (triggered by the loadFirst() below) writes it — without untrack,
    // that read makes diverseJobId a tracked dependency of THIS effect,
    // so the job-state write later on retriggers this whole effect,
    // which calls cancelDiverseJob() again mid-job and loops forever
    // (observed live: a real diverse job kept getting cancelled and
    // immediately restarted, 409ing against its own previous attempt).
    untrack(() => void cancelDiverseJob());
    diverse.diverseSelection = null;
    void loadFirst();
  });
  let filterDebounce: ReturnType<typeof setTimeout> | null = null;
  // Guards against a spurious extra firing of this effect re-running the
  // same query it just ran (observed live: a diverse-selection job could
  // get cancelled and immediately re-requested against itself, 409ing,
  // when this effect's body executed twice for the same filter state —
  // Svelte may batch/replay an effect body more than once per logical
  // change). Comparing against the last key this effect actually acted on
  // makes the debounced refetch (and, critically, the diverse-mode
  // cancel/invalidate below) idempotent regardless of how many times the
  // body runs for the same values.
  let lastFilterKey: string | null = null;
  $effect(() => {
    void sourceFilter;
    void slotTextQuery;
    void confMin;
    void confMax;
    // Strategy-bar sort/filter changes join the debounced path, not the
    // immediate one above — a sort pick or a threshold nudge shouldn't
    // feel snappier than dragging a confidence slider, and it keeps this
    // as the single refetch path new filters join (no third debounce).
    // strategyBar.k joins here too (P2-10) — a diverse-selection POST can
    // spawn a real backend job, so a k-stepper nudge must debounce the
    // same as everything else, not re-run per keystroke.
    void strategyBar.sort;
    void strategyBar.minMistakenness;
    void strategyBar.hideNearDuplicates;
    void strategyBar.k;
    void activeEnumParams;
    void activeUrlFilters;
    void itemFilterQuery;
    if (waitingForVocabulary) return;
    const key = JSON.stringify([
      sourceFilter,
      slotTextQuery,
      confMin,
      confMax,
      strategyBar.sort,
      strategyBar.minMistakenness,
      strategyBar.hideNearDuplicates,
      strategyBar.k,
      activeEnumParams,
      activeUrlFilters,
      itemFilterQuery,
    ]);
    // The first run only records the starting filters: the immediate
    // effect above already loads page 1, and fetching it a second time
    // 250ms later reset the cursor (breaking ?crop_id= deep links).
    if (lastFilterKey === null) {
      lastFilterKey = key;
      return;
    }
    if (filterDebounce) clearTimeout(filterDebounce);
    filterDebounce = setTimeout(() => {
      filterDebounce = null;
      if (key === lastFilterKey) return; // no real change — skip re-fetching
      lastFilterKey = key;
      // Any of the above changing invalidates a prior diverse selection
      // (new sort, new k, new term filter) — force runDiverseSelection()
      // to re-run on the next fetchPage rather than reusing a stale pool.
      // untrack() here for the same reason as the effect above: this
      // callback still runs inside the *effect's* reactive context (it's
      // synchronously reachable from the tracked $effect body via the
      // closure), so an untracked read of diverseJobId is still needed
      // to avoid diverseJobId writes re-triggering this effect.
      untrack(() => void cancelDiverseJob());
      diverse.diverseSelection = null;
      void loadFirst();
    }, 250);
    return () => {
      if (filterDebounce) {
        clearTimeout(filterDebounce);
        filterDebounce = null;
      }
    };
  });

  const current = $derived<ReviewItem | null>(queue.items[cursor] ?? null);

  const emptyMessage = $derived(
    emptyQueueMessage({
      label: tabs.activeQueueLabel,
      description: reviewTabsVocabularyStore.descriptionFor(activeTabEndpointId) ?? null,
      sortFallbackReason,
      emptyReason,
      emptyState: reviewTabsVocabularyStore.emptyState,
      importedTab: effectiveTab === 'imported',
      datasetsAvailable: datasetsAvailability.available === true,
      filtersActive: clientFiltersActive,
    }),
  );
  // Tabs whose last load with no client filters came back empty — dimmed
  // in the tab strip with a "0" so an operator isn't sent into them
  // blind (R3). Only ever learned from a real served total.
  let emptyTabEndpoints = $state<Record<string, boolean>>({});
  $effect(() => {
    if (queue.loading || queue.error || diverseMode || searchModeActive) return;
    if (queue.loadedPages === 0 || clientFiltersActive) return;
    const id = activeTabEndpointId;
    const empty = queue.total === 0;
    if (untrack(() => emptyTabEndpoints[id]) !== empty) {
      emptyTabEndpoints = { ...untrack(() => emptyTabEndpoints), [id]: empty };
    }
  });

  return {
    queue,
    handledIds,
    queueController,
    get cursor() {
      return cursor;
    },
    set cursor(v: number) {
      cursor = v;
    },
    get current() {
      return current;
    },
    get sortFallbackReason() {
      return sortFallbackReason;
    },
    get emptyReason() {
      return emptyReason;
    },
    get sortApplied() {
      return sortApplied;
    },
    maybePrefetch,
    get semanticSearchAvailable() {
      return semanticSearchAvailable;
    },
    get searchScores() {
      return searchScores;
    },
    applySearchResults,
    clearSearch,
    get currentPosition() {
      return currentPosition;
    },
    loadFirst,
    get emptyMessage() {
      return emptyMessage;
    },
    get emptyTabEndpoints() {
      return emptyTabEndpoints;
    },
  };
}

export type ReviewQueueLoader = ReturnType<typeof createReviewQueueLoader>;
