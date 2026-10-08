<script lang="ts">
  import { apiErrorText } from '$lib/api';
  import ItemFilterBar from '$lib/components/itemFilter/ItemFilterBar.svelte';
  import ServedFilterField from '$lib/components/itemFilter/ServedFilterField.svelte';
  import {
    ItemFilterState,
    withoutOpenVocab,
  } from '$lib/itemFilter/itemFilterState.svelte';
  import { ITEM_FILTER_PARAMS } from '$lib/itemFilter/itemFilterControls';
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import {
    cancelSelect,
    reviewUndismissCrop,
    getCrop,
    getCrops,
    getReviewQueue,
    getSelectStatus,
    getThumbUrl,
    locateInReviewQueue,
    selectDiverse,
    patchSlotMeta,
  } from '$lib/api';
  import { trapFocus } from '$lib/actions/trapFocus';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import BlurSlider from '$lib/components/BlurSlider.svelte';
  import CropMetaPanel from '$lib/components/CropMetaPanel.svelte';
  import ProvenanceChip from '$lib/components/ProvenanceChip.svelte';
  import MultiBoxCanvas from '$lib/components/MultiBoxCanvas.svelte';
  import VectorRefreshNotice from '$lib/components/review/VectorRefreshNotice.svelte';
  import RejectedBoxChips from '$lib/components/review/RejectedBoxChips.svelte';
  import { createMultiBoxRegionController } from '$lib/review/multiBoxRegionController.svelte';
  import ScoreChip from '$lib/components/ScoreChip.svelte';
  import ScrollStrip from '$lib/components/ScrollStrip.svelte';
  import SourceImageOverlay from '$lib/components/SourceImageOverlay.svelte';
  import ShortcutsButton from '$lib/components/ShortcutsButton.svelte';
  import SemanticSearchBox from '$lib/components/SemanticSearchBox.svelte';
  import StrategyBar from '$lib/components/StrategyBar.svelte';
  import SubjectScopeToggle from '$lib/components/SubjectScopeToggle.svelte';
  import { pushUndo, removeUndo, popUndo, reinsertAt } from '$lib/review/slotQueueOps';
  import { AbortRegistry } from '$lib/review/abortRegistry';
  import { createReviewQueueController } from '$lib/review/reviewController.svelte';
  import { buildSlotKeymap, rejectKeyGlyph } from '$lib/review/slotKeymap';
  import { isSlotSuppressedTab } from '$lib/review/slotTabGuard';
  import {
    emptyQueueMessage,
    locateMissMessage,
    NO_CLASS_YET,
    queuePosition,
    vlmEmptyReasonText,
  } from '$lib/review/reviewCopy';
  import { NO_OPINION_TEXT, probeOpinion } from '$lib/review/probeOpinion';
  import {
    humanWritableStates,
    statusWantsRejectionReason,
    panelLabels,
  } from '$lib/review/slotPanel';
  import { slotOf } from '$lib/annotations/cropSlots';
  import {
    itemClassTargets,
    itemHintClassIds,
    quickAssignClasses,
    resolveConfirmClassId,
    searchClasses,
  } from '$lib/classPicker';
  import {
    endpointForTab,
    isSlotTab,
    REVIEW_PRESETS,
    REVIEW_TABS,
    resolveEffectiveTab,
    tabHonorsPinnedSortDefault,
    visibleReviewTabs,
    type ReviewPresetId,
    reviewDeepLink,
    unavailableTabMessage,
  } from '$lib/reviewTabs';
  import { REGION_TAB_ID } from '$lib/annotations/servedRegionSlot';
  import { regionProfileStore } from '$stores/regionProfile.svelte';
  import { isDiverseOverlayAvailable } from '$lib/strategies';
  import type {
    Crop,
    DiverseSelection,
    RegistryClass,
    ReviewItem,
    ReviewTab,
  } from '$lib/types';
  import { createPager } from '$lib/pager.svelte';
  import { capCropDisplayStyle } from '$lib/review/cropDisplaySize';
  import { createStrategyBar } from '$lib/strategyBar.svelte';
  import { isSemanticSearchAvailable } from '$lib/strategies';
  import { subscribeCurationEvents, type CurationEventSubscription } from '$lib/sse';
  import { untrack } from 'svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import { keyboardStore, type RegisterActionOptions } from '$stores/keyboard.svelte';
  import { keymapStore } from '$stores/keymap.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { regionStatusesStore, toneRingRgb } from '$stores/regionStatuses.svelte';
  import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';
  import EmptyQueueEmbed from '$components/embedding/EmptyQueueEmbed.svelte';
  import { curationSettingsStore } from '$stores/curationSettings.svelte';
  import { undoStore } from '$stores/undo.svelte';
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { replaceState } from '$app/navigation';

  // Unified review by default — one continuous queue of every crop that
  // needs a human, sorted most-uncertain first. Down to 5 top-level tabs
  // (2026-09 consolidation, see $lib/reviewTabs.ts) — Mismatches / VLM
  // Low-Conf / Primary·Low-Conf collapsed into preset chips on the All
  // tab (below); Outliers retired entirely (see reviewTabs.ts doc
  // comment). The remaining 4 narrower tabs stay available for
  // diagnosing where uncertainty came from — each is a real, distinct
  // signal, not a rebrand of "everything." `new_class_proposals` (added
  // 2026-09-24, logic-moves W5) is a 6th real tab, not a preset — a
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
  // W10 is served; probe lazily, never on tabs that don't need it.
  $effect(() => {
    if (tab === 'imported') void datasetsAvailability.init();
  });
  // A `?tab=imported` link against a backend whose vocabulary lacks the tab
  // falls back to All once the vocabulary has loaded.
  $effect(() => {
    if (!reviewTabsVocabularyStore.loaded || tab !== 'imported') return;
    if (reviewTabsVocabularyStore.hasEntry('imported')) return;
    tab = 'all';
    unavailableTabNotice = unavailableTabMessage(
      'imported',
      REGION_TAB_ID,
      regionProfileStore.configured,
      regionProfileStore.unknown,
    );
  });
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
    const url = new URL(page.url);
    if (preset) {
      url.searchParams.set('preset', preset);
    } else {
      url.searchParams.delete('preset');
    }
    replaceState(resolve(projectHref(`/review${url.search}`)), {});
  }
  const pageSize = 30;
  let cursor = $state<number>(0); // index within accumulated items

  // Sort/filter strategy (curation-strategy plan Phase 3). 'default'
  // keeps every request byte-identical to pre-Phase-3 behavior — the
  // regression guard the backend plan requires (§8.6). 'diverse' is
  // registered as an overlay id (P2-10) so toQueryParams() never forwards
  // `sort=diverse` to {API_PREFIX}/review/{tab}, which 400s on it — diverse mode
  // is a wholly separate call (selectDiverse), not a sort param.
  const strategyBar = createStrategyBar({ overlayIds: ['diverse'] });
  // S1 (visual audit 2026-09-24, corrected status): needed so StrategyBar
  // can tell the operator when the deployment's pinned review-sort
  // default has no coverage yet. init() never throws and is idempotent —
  // safe to call unconditionally alongside strategiesStore's own $effect.
  $effect(() => {
    void curationSettingsStore.init();
  });
  // The deployment's pinned `sort` default, only when the active tab has
  // no tuned default of its own (see `tabHonorsPinnedSortDefault`) — every
  // other tab ignores the pinned default entirely, so passing it there
  // would be misleading even if StrategyBar's own gating would no-op it.
  const pinnedSortId = $derived(
    tabHonorsPinnedSortDefault(effectiveTab)
      ? (curationSettingsStore.settings.defaults.sort ?? null)
      : null,
  );
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

  // -- diverse overlay (P2-10, pool-scale k-center-greedy selection) ----
  // Entered via the same StrategyBar sort dropdown as every other sort —
  // picking 'diverse' swaps the queue's item source entirely (a one-
  // item-at-a-time cursor has no "side panel" to put an overlay in).
  const diverseAvailable = $derived(
    isDiverseOverlayAvailable(strategiesStore.methods.overlays),
  );
  const diverseMode = $derived(strategyBar.sort === 'diverse' && diverseAvailable);
  const DIVERSE_K_DEFAULT = 100;
  // The pool selection job's own served cap (`select_max_k`); none = no clamp.
  const diverseKMax = $derived(
    strategiesStore.methods.overlays.find((o) => o.id === 'diverse')?.select_max_k ??
      undefined,
  );
  let diverseSelection = $state<DiverseSelection | null>(null);
  let diverseJobId = $state<string | null>(null);
  let diverseJobStatus = $state<string | null>(null);
  let diverseError = $state<string | null>(null);
  let diversePoll: ReturnType<typeof setInterval> | null = null;

  function stopDiversePolling(): void {
    if (diversePoll) clearInterval(diversePoll);
    diversePoll = null;
  }

  function termFilters(): Record<string, unknown> {
    // Server-side, scope.filters on POST {API_PREFIX}/select/diverse only supports
    // term/terms filters (class_name, source) — NOT conf_min/conf_max/
    // min_blur_ratio/max_rank/region text. Those controls are disabled in
    // the UI while diverseMode is active (see the filter bar below) so
    // this never silently drops something the operator thinks is applied.
    const f: Record<string, unknown> = {};
    if (sourceFilter) f.source = sourceFilter;
    // scope.filters are exact-match terms keyed by index field; a list
    // becomes a `terms` clause.
    if (itemFilter.classNames.length > 0) f.class_name = [...itemFilter.classNames];
    return f;
  }

  async function runDiverseSelection(): Promise<DiverseSelection | null> {
    stopDiversePolling();
    diverseError = null;
    diverseJobId = null;
    diverseJobStatus = null;
    const k = strategyBar.k ?? DIVERSE_K_DEFAULT;
    const res = await selectDiverse(
      { review_tab: endpointForTab(effectiveTab), filters: termFilters() },
      k,
    );
    if (res.kind === 'disabled') {
      diverseError = 'Diverse selection is disabled on the backend.';
      return null;
    }
    if (res.kind === 'already_running') {
      diverseError =
        'A diverse-selection job is already running (singleton — another operator or tab may be using it). Try again shortly.';
      return null;
    }
    if (res.kind === 'ready') {
      diverseSelection = res.selection;
      return res.selection;
    }
    // 202 — large pool (the `all` tab always takes this path, ~320k
    // crops). Poll until done, mirroring
    // /routes/bakeoff/+page.svelte's setInterval pattern. CONFIRMED
    // against the real backend (selection/job.py's _JobState): status is
    // 'running' | 'completed' | 'failed' | 'cancelled', and the finished
    // selection is nested under `result` — not flattened onto the status
    // object.
    diverseJobId = res.job_id;
    diverseJobStatus = 'running';
    return new Promise((resolve) => {
      diversePoll = setInterval(async () => {
        try {
          const st = await getSelectStatus();
          diverseJobStatus = st.status;
          if (st.status === 'running') return;
          stopDiversePolling();
          if (st.status === 'completed' && st.result) {
            diverseSelection = st.result;
            diverseJobId = null;
            diverseJobStatus = null;
            resolve(st.result);
            return;
          }
          // 'failed' | 'cancelled' | any unrecognized terminal status —
          // clear diverseJobId too, not just the poll interval, so the
          // "selecting… (failed) cancel" chip doesn't linger for a job
          // that already terminated server-side (real backend behavior
          // observed live: k_center_greedy's tight numpy loop blocks the
          // event loop long enough that the heartbeat goes stale and
          // get_state() reports 'failed' mid-computation, even though the
          // job goes on to finish and overwrite its own state to
          // 'completed' moments later — a OpenProcessor timing quirk, out of
          // scope to fix here, but the frontend must still stop treating
          // this job as "ours" once it reports terminal).
          diverseError = st.error ?? `Diverse selection ${st.status}.`;
          diverseJobId = null;
          diverseJobStatus = null;
          resolve(null);
        } catch (e) {
          stopDiversePolling();
          diverseError = `Diverse selection failed: ${apiErrorText(e)}`;
          diverseJobId = null;
          diverseJobStatus = null;
          resolve(null);
        }
      }, 2000);
    });
  }

  async function cancelDiverseJob(): Promise<void> {
    stopDiversePolling();
    if (diverseJobId) {
      try {
        await cancelSelect();
      } catch {
        /* best-effort — the job may have already finished */
      }
    }
    diverseJobId = null;
    diverseJobStatus = null;
  }

  /**
   * Diverse mode replaces the queue's item source entirely — the pager's
   * fetchPage slices `diverseSelection.crop_ids` into page_size chunks
   * and hydrates each id via getCrop, since {API_PREFIX}/select/diverse only
   * returns ids, not full crop records. Each hydrated crop already
   * carries its own served `proposed_class_id`/`_name` (item 11,
   * 2026-09-24 logic-moves — promoted onto `Crop`/`mapRawCrop`, so
   * `getCrop` returns it same as `getReviewQueue`) — no client fill-in
   * needed. resolveConfirmClassId/canConfirm falls back to the crop's
   * existing class_id when it's null (classPicker.ts), so Enter still
   * does the right thing either way: confirms the proposal or existing
   * label if either exists, otherwise opens the class picker.
   */
  async function fetchDiversePage(
    page: number,
  ): Promise<{ items: ReviewItem[]; total: number } | null> {
    let selection = diverseSelection;
    if (!selection) {
      selection = await runDiverseSelection();
      if (!selection) return { items: [], total: 0 };
    }
    const start = (page - 1) * pageSize;
    const ids = selection.crop_ids.slice(start, start + pageSize);
    const crops = await Promise.all(ids.map((id) => getCrop(id)));
    const items: ReviewItem[] = crops.map((crop) => ({
      ...crop,
      reason: 'diverse selection (k-center-greedy)',
    }));
    return { items, total: selection.crop_ids.length };
  }

  // Queue pager. One fetchPage closure means the tab + filter set can't
  // drift between page 1 and the pages the cursor pulls in behind it.
  const queue = createPager<ReviewItem>({
    fetchPage: async (page) => {
      if (diverseMode) return fetchDiversePage(page);
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
  let searchScores = $state(new Map<string, number>());

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
  // actions and arrow-key nav below) and are handed in by
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

  // Filter bar. `sourceFilter` sends `source` to {API_PREFIX}/review/{tab} (item
  // 14/G3, 2026-09-24 logic-moves — renamed off the old `hdd_source`
  // control, which the endpoint never actually read). `termFilters()`
  // below (the diverse-selection scope, a different endpoint) now sends
  // the same value under `source` too — the OpenProcessor 1327181 naming
  // sweep (F9) removed `?hdd_source=` outright, so both call sites agree.
  let sourceFilter = $state<string>('');
  // The shared item filter (class by name, origin, embedding / review state,
  // area band), seeded from the URL and persisted back to it. Conf and the
  // largest-N toggle keep their own controls below; the bar never draws them.
  const itemFilter = new ItemFilterState();
  itemFilter.fromUrl(page.url.searchParams);
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
    const url = new URL(page.url);
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
  // changes (see the immediate `$effect` below); seeded from the URL on
  // first load so `?region_status=verify_rejected` is bookmarkable, the
  // same pattern `preset` uses.
  const NON_ENUM_FILTER_PARAMS = new Set([
    'tab',
    'preset',
    'crop_id',
    'import_id',
    'combine_conflict',
    ...ITEM_FILTER_PARAMS,
  ]);
  type ServedFilterValue = string | string[];
  let enumFilterValues = $state<Record<string, ServedFilterValue>>(
    Object.fromEntries(
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
  // URL-seeded non-enum filters (W10 `import_id`, P4 `combine_conflict`):
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
    const url = new URL(page.url);
    url.searchParams.delete(param);
    replaceState(resolve(projectHref(`/review${url.search}`)), {});
  }
  function setEnumFilter(param: string, value: ServedFilterValue): void {
    enumFilterValues = { ...enumFilterValues, [param]: value };
    const url = new URL(page.url);
    url.searchParams.delete(param);
    for (const v of Array.isArray(value) ? value : value ? [value] : []) {
      url.searchParams.append(param, v);
    }
    replaceState(resolve(projectHref(`/review${url.search}`)), {});
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

  // `/review?crop_id=` deep link (item 10, 2026-09-24 logic-moves W5):
  // ask the backend exactly where the crop sits under the active tab's
  // filters/sort via GET {API_PREFIX}/review/{tab}/locate, rather than the old
  // approach of blindly paging forward up to 300 items hoping to find
  // it. `in_queue: false` means it doesn't match this tab (already
  // handled, filtered out, etc.) — `reason` explains why when the
  // backend sends one.
  let jumpingToCrop = $state(false);
  async function jumpToPendingCrop(): Promise<void> {
    const cropId = pendingCropId;
    if (cropId == null) return;
    if (diverseMode || searchModeActive) {
      // Neither a pool-scale overlay selection nor a semantic-search
      // result set has a stable server-side "locate" — drop the deep
      // link rather than spin forever waiting for a match that can
      // never resolve.
      pendingCropId = null;
      awaitingDeepLink = false;
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
      pendingCropId = null;
      awaitingDeepLink = false;
    }
  }

  $effect(() => {
    if (pendingCropId == null || queue.loading || queue.loadingMore || jumpingToCrop) {
      return;
    }
    if (queue.loadedPages === 0) return; // first page not in yet
    void jumpToPendingCrop();
  });

  $effect(() => {
    keyboardStore.setScope('review');
  });

  // -- SSE: live updates as ingest + workers classify new crops -------
  // We don't fetch each new crop individually (no single-crop endpoint
  // is exposed); instead we count incoming crop.* events and surface a
  // "X new crops" pill the user can click to refresh the queue. Auto-
  // refreshing while the operator is mid-keystroke would be jarring —
  // they decide when to pull in the new batch.
  let liveNewCount = $state<number>(0);
  let liveSub: CurationEventSubscription | null = null;
  onMount(() => {
    liveSub = subscribeCurationEvents({
      // Both classification + any slot-verify changes are interesting on
      // the review page — the operator may be on any tab. `_verified` is
      // structural (matches every crop.<slot.key>_verified event, plus
      // the generic crop.region_verified — see
      // sse.ts's slotVerifiedEventTypes()), not a single hardcoded slot.
      onEvent: (ev) => {
        if (
          ev.type === 'crop.classified' ||
          ev.type === 'crop.created' ||
          ev.type.endsWith('_verified')
        ) {
          liveNewCount += 1;
        }
      },
    });
    return () => {
      liveSub?.close();
      liveSub = null;
    };
  });

  function refreshFromLive(): void {
    liveNewCount = 0;
    void loadFirst();
  }

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
    diverseSelection = null;
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
      diverseSelection = null;
      void loadFirst();
    }, 250);
    return () => {
      if (filterDebounce) {
        clearTimeout(filterDebounce);
        filterDebounce = null;
      }
    };
  });

  onMount(() => stopDiversePolling);

  const current = $derived<ReviewItem | null>(queue.items[cursor] ?? null);
  // A slot that declares `listField` (the served region slot, always) uses
  // MultiBoxCanvas/multiBoxRegionController. A tier-2 slot with only a
  // scalar `bboxField` is display-only: the backend has no write route for
  // it, so it has no edit mode.
  const isMultiBoxSlot = $derived(activeSlot?.capabilities.subBox?.listField != null);
  // 3f1a11e adoption, updated for W8: the authoritative, kind-styled
  // explanation for the "Reason" row — the generic per-item `reason`
  // string (see below) always says "verifier rejected this candidate
  // (…)", wrong wording for a needs_human item. W8 moved the MACHINE
  // reason off the item (`region_rejection_reason` is now the reviewer's
  // free-text note only, per spec) onto each rejected box
  // (`SlotBox.rejectionReason`) — for a multi-box slot this reads the
  // first rejected box's reason; a tier-2 single-box slot still reads
  // the item-level field. Only falls back to `current.reason` when
  // neither is present (core tabs, e.g. the mismatches preset).
  // W10: the served per-box `locked` flag, by box id, for the canvas's
  // lock glyph (the editable working set carries no `locked`).
  const lockedBoxIds = $derived<Set<string>>(
    new Set(
      (activeSlot && current ? (slotOf(current, activeSlot)?.subBoxes ?? []) : [])
        .filter((b) => b.locked === true && b.boxId != null)
        .map((b) => b.boxId as string),
    ),
  );
  const currentSlotRejectionReason = $derived<string | null>(
    activeSlot && current
      ? isMultiBoxSlot
        ? (slotOf(current, activeSlot)?.subBoxes?.find((b) => b.state === 'rejected')
            ?.rejectionReason ?? null)
        : (slotOf(current, activeSlot)?.lifecycle?.rejectionReason ?? null)
      : null,
  );
  // DQ-M8: served role (classSourcesStore, GET {API_PREFIX}/class_sources), not
  // a hardcoded string match — mirrors sourceBadge.ts's
  // role.startsWith('vlm') check for the label-source badge.
  const isCurrentLabelVlmSourced = $derived(
    (classSourcesStore.roleFor(current?.label_source) ?? '').startsWith('vlm'),
  );

  // R1 (visual audit 2026-09-24): item-class targets only — a slot-bound
  // region class is never an item's class — ranked by what THIS item
  // already points at (proposal / current / VLM / model), then by
  // validated count, instead of by global validated count alone.
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
  const emptyMessage = $derived(
    emptyQueueMessage({
      label: activeQueueLabel,
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
  const currentHintIds = $derived(itemHintClassIds(current));
  const topClasses = $derived(
    quickAssignClasses(classesStore.classes, currentHintIds, 10),
  );
  const pickerClasses = $derived(itemClassTargets(classesStore.classes));

  // P1-5: what Enter/Confirm would actually assign, or null when there's
  // nothing to confirm (67/100 `all`-tab items today). Drives the Confirm
  // button's disabled state and whether Enter confirms vs. opens the class
  // picker below.
  const canConfirm = $derived(resolveConfirmClassId(current) != null);

  // -- class picker (P1-4) ---------------------------------------------
  // Fuzzy-search combobox over *every* non-deprecated class, opened with
  // '/'. topClasses above caps quick-assign at the 10 most-validated
  // classes; 73 of 84 need a round-trip to /classes without this. Pure
  // filter/ranking logic lives in $lib/classPicker.ts (searchClasses) so
  // it's unit-testable without @testing-library/svelte.
  let pickerOpen = $state(false);
  let pickerQuery = $state('');
  let pickerIndex = $state(0);
  let pickerInputEl = $state<HTMLInputElement | null>(null);

  const pickerResults = $derived(
    searchClasses(pickerClasses, pickerQuery, 50, currentHintIds),
  );

  // Re-center the highlighted row on the best match whenever the query
  // (re-ranks the list) changes. Doesn't fire on arrow-key navigation,
  // which only touches pickerIndex.
  $effect(() => {
    void pickerQuery;
    pickerIndex = 0;
  });

  function openPicker(): void {
    if (isSlotTab(tab) || !current) return;
    pickerOpen = true;
    pickerQuery = '';
    pickerIndex = 0;
    // Input isn't in the DOM until this render commits.
    requestAnimationFrame(() => pickerInputEl?.focus());
  }

  function closePicker(): void {
    pickerOpen = false;
    pickerQuery = '';
    pickerIndex = 0;
  }

  async function pickClass(cls: RegistryClass): Promise<void> {
    closePicker();
    await assign(cls.id);
  }

  // Element-scoped handler on the picker's own <input> — not a second
  // window keydown listener (keyboard.svelte.test.ts's guard). Browser
  // focus already keeps this from colliding with keyboardStore/the
  // layout's per-class dispatcher: both treat a focused <input> as a
  // typing target and skip it entirely.
  function onPickerKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      e.preventDefault();
      closePicker();
      return;
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      pickerIndex = Math.min(pickerResults.length - 1, pickerIndex + 1);
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      pickerIndex = Math.max(0, pickerIndex - 1);
      return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      const cls = pickerResults[pickerIndex];
      if (cls) void pickClass(cls);
    }
  }

  /**
   * Optimistically drop an item from the queue and advance. Thin wrapper
   * around the controller's `removeFromQueue` — kept under this name
   * because slot actions (confirmSlot/rejectSlot/markFalsePositive) below
   * still call it directly.
   */
  function _removeFromQueue(item: ReviewItem): () => void {
    return queueController.removeFromQueue(item);
  }

  async function assign(classId: number): Promise<void> {
    if (!current) return;
    await queueController.assign(current, classId);
  }

  async function confirmAndAdvance(): Promise<void> {
    if (!current) return;
    const proposed = resolveConfirmClassId(current);
    if (proposed == null) {
      // Enter's registration below already routes here vs. openPicker()
      // based on canConfirm, so this only fires from the Confirm button —
      // which is disabled in this state — or a stale click race. Keep the
      // toast as a safety net either way.
      toastStore.warn(
        `No proposed class on this item — press ${kg('review.queue.class_picker')} to search.`,
      );
      return;
    }
    await assign(proposed);
  }

  /** "Accept model's class" (item 14, 2026-09-24 logic-moves): the
   *  model_disagreements tab's probe prediction now carries its own
   *  `probe_pred_class_id`, so this assigns it directly — no name→id
   *  lookup needed (closes G4). */
  // F8 D1: the served probe opinion decides whether a prediction is
  // shown and whether "Accept model's class" is offered.
  const opinion = $derived(
    current ? probeOpinion(current) : { kind: 'none' as const, showAccept: false },
  );

  async function acceptModelClass(): Promise<void> {
    if (!current || current.probe_pred_class_id == null || !opinion.showAccept) return;
    await assign(current.probe_pred_class_id);
  }

  function skip(): void {
    queueController.skip();
  }

  async function discard(): Promise<void> {
    if (!current) return;
    // Discard = "permanently dismiss this crop from every review queue."
    // Stamps review_dismissed_at on the backend; the review queue's
    // must_not filter excludes any crop with that field set. The crop's
    // class / region state is left intact — this is NOT an unlabel. See
    // reviewController.svelte.ts's `discard` for the undo-entry note.
    await queueController.discard(current);
  }

  // -- dismissed-crops panel (un-dismiss) --------------------------------
  // Minimal reachable-from-/review surface for reviewUndismissCrop: a
  // toggleable panel listing crops discard() has dismissed, each with a
  // Restore action. Loaded lazily (only when opened), not kept in sync
  // with the live queue.
  let dismissedPanelOpen = $state<boolean>(false);
  let dismissedItems = $state<Crop[]>([]);
  let dismissedLoading = $state<boolean>(false);
  let dismissedError = $state<string | null>(null);

  // -- item-detail "Details" disclosure (G7/G9/G8) -----------------------
  // Collapsed by default; CropMetaPanel only mounts (and fetches
  // history/image) once the operator opens it.
  let detailsOpen = $state<boolean>(false);

  async function toggleDismissedPanel(): Promise<void> {
    dismissedPanelOpen = !dismissedPanelOpen;
    if (!dismissedPanelOpen) return;
    dismissedLoading = true;
    dismissedError = null;
    try {
      // M1 (2026-09-24 interactive pass): GET {API_PREFIX}/crops's `sort`
      // is a '<field>[:asc|desc]' pair against a closed field list
      // (contracts/openprocessor/openapi/curation.json) — 'recent' isn't
      // one of them and 400s every time. 'updated_at:desc' is the
      // server's own documented default and matches "most recently
      // dismissed first".
      const res = await getCrops({
        review_dismissed: true,
        limit: 60,
        sort: 'updated_at:desc',
      });
      dismissedItems = res.items;
    } catch (e) {
      // Show the failure inline instead of falling through to "No
      // dismissed crops" — that empty state used to render even when the
      // request itself failed, making a real 400 look like there was
      // simply nothing to restore.
      dismissedError = apiErrorText(e);
      dismissedItems = [];
    } finally {
      dismissedLoading = false;
    }
  }

  async function undismiss(crop: Crop): Promise<void> {
    try {
      await reviewUndismissCrop(crop.id);
      dismissedItems = dismissedItems.filter((c) => c.id !== crop.id);
      toastStore.success('Restored to review.');
    } catch (e) {
      toastStore.error(`Restore failed: ${apiErrorText(e)}`);
    }
  }

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

  // Every key this page prints comes from the keymap (never a literal).
  const kg = (actionId: string) => keymapStore.glyph(actionId);
  const undoHint = () =>
    `Press ${kg('review.undo')} to undo, step back with ${kg('review.region.back')}.`;

  // W8 multi-box: per-box state → ring color/dash. The ring color now
  // reads the served box_states `tone` (backend follow-up to W8.7,
  // feat/w8-multibox-lockstep) via `toneRingRgb(boxStateTone(state))` —
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
    cursor = insertAt;
    toastStore.info(
      `Stepped back. Press ${kg('review.region.edit_box')} to re-edit, ${kg('review.region.confirm')} to re-confirm.`,
    );
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

  /** Confirm for a slot with no box list: a status-only write. */
  async function confirmSlot(): Promise<void> {
    if (!current || !activeSlot) return;
    const confirmStatus =
      regionStatusesStore.confirmStatus ??
      activeSlot.capabilities.lifecycle?.confirmState;
    if (!confirmStatus) return;
    const item = current;
    const undoEntry: SlotUndoEntry = { item, insertAt: cursor };
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
   * W8 multi-box Enter (owner decision): confirms only `proposed` boxes,
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
    const undoEntry: SlotUndoEntry = { item, insertAt: cursor };
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
    const undoEntry: SlotUndoEntry = { item, insertAt: cursor };
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

  async function undoLast(): Promise<void> {
    // `crop` (from POST {API_PREFIX}/crops/{id}/label/undo or
    // .../label/undo_batch) already carries its own served
    // proposed_class_id/_name (item 11, 2026-09-24 logic-moves) — no
    // client fill-in. `reason` is the only field the controller adds;
    // every other queue-only field has no meaningful value for a
    // restored item.
    await queueController.undoLast();
  }

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
          cursor = Math.min(queue.items.length - 1, cursor + 1);
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
        // W8 per-box actions (y/r/Tab) — reserved letters, see
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
        if (!slotCanvas) return;
        const target = e.target as HTMLElement | null;
        if (target && /^(input|textarea|select)$/i.test(target.tagName)) return;
        if (slotCanvas.handleKey(e)) e.preventDefault();
      };
      window.addEventListener('keydown', canvasKey);
    } else if (!isSlotTab(tab)) {
      // On non-slot tabs arrow keys navigate the queue.
      reg('review.queue.prev', () => {
        cursor = Math.max(0, cursor - 1);
      });
      reg('review.queue.next', () => {
        cursor = Math.min(queue.items.length - 1, cursor + 1);
        maybePrefetch();
      });
    }

    return () => {
      offs.forEach((off) => off());
      if (canvasKey) window.removeEventListener('keydown', canvasKey);
    };
  });
</script>

<div class="flex h-full flex-col">
  <!-- Tabs — horizontally scrollable on narrow viewports so all tabs stay reachable
       without colliding with the loaded-count chip on the right. -->
  <div class="flex items-center gap-1 border-b border-zinc-800 px-4">
    <!-- R2 (visual audit 2026-09-24): at 800px the plain overflow row
         clipped "New class proposals"/"Regions" with no hint, and an
         active tab past the edge was invisible — ScrollStrip shows a
         chevron where tabs are hidden and scrolls the active one in. -->
    <ScrollStrip class="gap-1" activeKey={tab} testId="review-tabs">
      {#each visibleTabs as t (t.id)}
        {@const knownEmpty = emptyTabEndpoints[t.endpointId] === true}
        <button
          type="button"
          data-active={tab === t.id ? 'true' : undefined}
          class="shrink-0 px-3 py-2.5 text-sm border-b-2 {tab === t.id
            ? 'border-blue-500 text-white'
            : knownEmpty
              ? 'border-transparent text-zinc-600 hover:text-zinc-300'
              : 'border-transparent text-zinc-400 hover:text-zinc-200'}"
          title={[
            reviewTabsVocabularyStore.descriptionFor(t.endpointId),
            knownEmpty ? 'Empty when last loaded.' : null,
          ]
            .filter(Boolean)
            .join(' — ') || undefined}
          onclick={() => {
            tab = t.id;
            pendingCropId = null;
            awaitingDeepLink = false;
            const url = new URL(page.url);
            url.searchParams.set('tab', t.urlId);
            url.searchParams.delete('crop_id');
            url.searchParams.delete('preset');
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
            replaceState(resolve(projectHref(`/review${url.search}`)), {});
            // Presets only make sense on the All tab — switching to any
            // other tab (or re-landing on All from one) always starts
            // from plain All rather than silently carrying a stale chip.
            preset = null;
            closePicker();
          }}
        >
          {reviewTabsVocabularyStore.labelFor(t.endpointId, t.label)}
          {#if knownEmpty}
            <span
              class="ml-1 font-mono text-[10px] text-zinc-600"
              data-testid="tab-empty-count">0</span
            >
          {/if}
        </button>
      {/each}
    </ScrollStrip>
    {#if regionProfileStore.unknown}
      <!-- F-78: boot couldn't read the region profile yet; the region tab
           appears once a /health poll answers (the layout re-mounts). -->
      <span
        class="shrink-0 pl-2 text-[11px] text-zinc-500"
        data-testid="region-profile-loading">loading region profile…</span
      >
    {/if}
    <span
      data-testid="queue-counter"
      class="shrink-0 pl-2 font-mono text-xs text-zinc-500"
    >
      {queue.items.length > 0 ? `#${currentPosition}` : '—'} · {queue.items.length} loaded ·
      {queue.total}
      total
    </span>
    <span class="shrink-0 pl-2"><ShortcutsButton /></span>
    <button
      type="button"
      class="ml-2 shrink-0 rounded border border-zinc-700 px-2 py-1 text-[11px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
      onclick={toggleDismissedPanel}
      title="Crops permanently dismissed from review — restore one back into the queue"
    >
      Dismissed
    </button>
    {#if liveNewCount > 0}
      <button
        type="button"
        class="ml-2 shrink-0 animate-pulse rounded-full border border-blue-500/60 bg-blue-500/15 px-2.5 py-1 text-[11px] text-blue-200 hover:bg-blue-500/25"
        onclick={refreshFromLive}
        title="Reload the queue with the latest crops"
      >
        {liveNewCount} new · refresh
      </button>
    {/if}
  </div>

  <!-- Strategy bar — collapsed one-line sort/filter chip by default (see
       StrategyBar.svelte); the fallback note only appears when the
       server couldn't honor the requested sort. -->
  <div class="flex items-center gap-3 border-b border-zinc-800 px-4 py-1.5">
    <!-- Left-aligned, first control in the row, so it reads as the
         primary way in rather than a control squeezed after the strategy
         bar. -->
    {#if semanticSearchAvailable}
      <SemanticSearchBox
        filter={{ tab: endpointForTab(effectiveTab), ..._filter() }}
        pageSize={200}
        onResults={(res) => {
          searchModeActive = true;
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
        }}
        onClear={() => {
          searchModeActive = false;
          searchScores = new Map();
          void queue.loadFirst();
        }}
      />
    {/if}
    <StrategyBar
      bar={strategyBar}
      offerDiverse={diverseAvailable}
      appliedSort={sortApplied}
      fallbackReason={sortFallbackReason}
      {pinnedSortId}
      diverseKDefault={DIVERSE_K_DEFAULT}
      {diverseKMax}
      diverseMeta={diverseSelection
        ? {
            method: diverseSelection.method,
            version: diverseSelection.version,
            n_pool: diverseSelection.n_pool,
          }
        : null}
    />
    <!-- M11: sort_fallback_reason now renders inside StrategyBar's own
         summary chip, next to sort_applied, instead of a separate banner
         here — see fallbackReason above. -->
    {#if diverseMode}
      {#if diverseJobId}
        <span class="text-[11px] text-blue-300">
          selecting… ({diverseJobStatus ?? 'running'})
          <button
            type="button"
            class="ml-1 underline hover:text-blue-100"
            onclick={() => void cancelDiverseJob()}
          >
            cancel
          </button>
        </span>
      {/if}
      {#if diverseError}
        <span class="text-[11px] text-red-300" title={diverseError}>
          {diverseError}
        </span>
      {/if}
    {/if}
  </div>

  <!-- Filter bar — flex children keep their width via flex-shrink-0; hotkey hint
       hides below md so it doesn't collide with controls on narrow viewports
       (same content is on the ~ overlay). -->
  <div
    class="flex min-w-0 flex-wrap items-center gap-3 border-b border-zinc-800 bg-zinc-900/40 px-4 py-2 text-xs"
  >
    <!-- GET /review/{tab} accepts class_id/source/conf_min/conf_max as of
         the 2026-09-24 logic-moves cutover (item 14/G3) — re-enabled,
         server-side, unconditionally (no client-side filtering here). -->
    {#if filterVisible('source')}
      <label class="flex shrink-0 items-center gap-1.5">
        <span class="text-zinc-400">Source</span>
        <input
          type="text"
          bind:value={sourceFilter}
          placeholder="any"
          class="input-sm w-32"
        />
      </label>
    {/if}

    <ItemFilterBar
      state={itemFilter}
      visible={itemFilterVisible}
      served={servedSpecs}
      onchange={persistItemFilter}
      rootClass="contents"
    />

    {#if filterVisible('conf_min') || filterVisible('conf_max')}
      <label
        class="flex shrink-0 items-center gap-1.5"
        class:opacity-40={diverseMode}
        title={diverseMode ? 'not applied to diverse selection' : undefined}
      >
        <span class="text-zinc-400">Conf</span>
        <input
          type="number"
          min="0"
          max="1"
          step="0.05"
          bind:value={confMin}
          disabled={diverseMode}
          aria-label="Minimum confidence"
          class="input-sm w-16"
        />
        <span class="text-zinc-500">..</span>
        <input
          type="number"
          min="0"
          max="1"
          step="0.05"
          bind:value={confMax}
          disabled={diverseMode}
          aria-label="Maximum confidence"
          class="input-sm w-16"
        />
      </label>
    {/if}

    {#if activeSlot?.capabilities.queue?.textFilter && filterVisible(activeSlot.capabilities.queue.textFilter.param)}
      <label
        class="flex shrink-0 items-center gap-1.5"
        class:opacity-40={diverseMode}
        title={diverseMode ? 'not applied to diverse selection' : undefined}
      >
        <span class="text-zinc-400">{activeSlot.capabilities.queue.textFilter.label}</span
        >
        <input
          type="text"
          bind:value={slotTextQuery}
          disabled={diverseMode}
          placeholder={activeSlot.capabilities.queue.textFilter.placeholder}
          class="input-sm w-28"
        />
      </label>
    {/if}

    <!-- Generic served-enum filter bar (3f1a11e adoption) — one <select>
         per ReviewFilterSpec the active tab's GET {API_PREFIX}/review/tabs entry
         declares (e.g. Regions' region_status: all / detected only /
         verifier-rejected candidates only). No param-specific markup —
         a future spec on any tab renders here unchanged. -->
    {#each activeFilterSpecs as spec (spec.param)}
      <ServedFilterField
        {spec}
        value={enumFilterValues[spec.param]}
        onchange={setEnumFilter}
      />
    {/each}

    <!-- Always available, on every tab and preset — matches Conf/Class/HDD
         source below/above, and the backend's own query builder already
         treats max_rank/min_blur_ratio as tab-agnostic. Used to be gated to
         only primary_low_conf/classifier_blind_spots, which made these controls
         appear and disappear depending on which tab or chip was active.
         Disabled (not hidden) while diverseMode is active — scope.filters
         on POST {API_PREFIX}/select/diverse doesn't support max_rank/min_blur_ratio
         (P2-10), so applying either here would silently do nothing. -->
    <div
      class="contents"
      class:opacity-40={diverseMode}
      class:pointer-events-none={diverseMode}
      title={diverseMode ? 'not applied to diverse selection' : undefined}
    >
      {#if filterVisible('max_rank')}
        <SubjectScopeToggle
          bind:value={subjectScope}
          labels={[
            servedMaxRankDefault != null ? `Top ${servedMaxRankDefault}` : 'All ranks',
            'Largest',
            '+2nd',
          ]}
          titles={[
            servedMaxRankDefault != null
              ? `This tab's own default: only the ${servedMaxRankDefault} largest subjects in each image`
              : 'Every subject in the image, whatever its size',
            'Only the largest subject in each image',
            'The largest and second-largest subject in each image',
          ]}
          label="subject"
        />
      {/if}
      {#if filterVisible('min_blur_ratio')}
        <BlurSlider
          bind:value={blurSlider}
          oncommit={commitBlur}
          max={BLUR_MAX}
          title="Hide crops blurrier than this"
        />
      {/if}
    </div>

    {#if importIdFilter && urlFilterServed('import_id')}
      <button
        type="button"
        class="chip border-blue-500/60 bg-blue-500/15 text-blue-100"
        data-testid="filter-chip-import-id"
        title="Remove this filter"
        onclick={() => clearUrlFilter('import_id')}
      >
        Import <code class="font-mono">{importIdFilter}</code> ×
      </button>
    {/if}
    {#if combineConflictFilter && urlFilterServed('combine_conflict')}
      <button
        type="button"
        class="chip border-blue-500/60 bg-blue-500/15 text-blue-100"
        data-testid="filter-chip-combine-conflict"
        title="Remove this filter"
        onclick={() => clearUrlFilter('combine_conflict')}
      >
        Combine conflicts only ×
      </button>
    {/if}

    {#if tab === 'all'}
      <!-- Quick-filter preset chips (2026-09 tab consolidation) — Mismatches
           / VLM Low-Conf / Primary·Low-Conf collapsed from top-level tabs
           into these, since live counts showed each was too big (11-97% of
           the dataset) to be a curated queue. Each chip reuses that former
           tab's exact backend query unchanged (see $lib/reviewTabs.ts);
           radio-style — picking a second chip swaps the first, clicking the
           active one (or "clear") returns to plain All. -->
      <div class="flex shrink-0 flex-wrap items-center gap-1.5">
        <!-- m31 (2026-09-24 interactive pass): "Quick filter" implied
             each chip narrows All — Primary · low-conf alone returns
             374 rows, more than All's 114, because both apply different
             ranking/eligibility rules server-side, not a subset relation.
             "Queue:" doesn't claim either one is smaller. -->
        <span class="text-zinc-400">Queue:</span>
        {#each REVIEW_PRESETS as p (p.id)}
          <button
            type="button"
            class="chip {preset === p.id
              ? 'border-blue-500/60 bg-blue-500/15 text-blue-100'
              : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800'}"
            aria-pressed={preset === p.id}
            title={reviewTabsVocabularyStore.descriptionFor(p.id) ?? p.description}
            onclick={() => togglePreset(p.id)}
          >
            {reviewTabsVocabularyStore.labelFor(p.id, p.label)}
          </button>
        {/each}
        {#if preset}
          <button
            type="button"
            class="chip bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
            onclick={() => {
              if (preset) togglePreset(preset);
            }}
          >
            clear
          </button>
        {/if}
      </div>
    {/if}

    <span class="grow"></span>

    <span class="hidden text-[11px] text-zinc-500 md:inline">
      {#if isMultiBoxSlot && activeSlot}
        <kbd>{kg('review.region.confirm')}</kbd> confirm proposed ·
        <kbd>{kg('review.region.accept_box')}</kbd> accept box ·
        <kbd>{kg('review.region.reject_box')}</kbd> reject box ·
        <kbd>{kg('box_edit.next_box')}</kbd> next box
        {#if editMode}
          · <kbd>{kg('box_edit.delete_box')}</kbd> delete box · drag empty area to add
        {:else}
          · <kbd>{kg('review.region.edit_box')}</kbd> edit
        {/if}
        · <kbd>{kg('review.skip')}</kbd> skip
      {:else if activeSlot}
        <kbd>{kg('review.region.confirm')}</kbd> confirm ·
        <kbd>{rejectKeyGlyph(activeSlot)}</kbd> reject
        {#if activeSlot.capabilities.lifecycle?.falsePositiveState}
          · <kbd>{kg('review.region.false_positive')}</kbd> false-pos
        {/if}
        · <kbd>{kg('review.skip')}</kbd> skip · <kbd>{kg('review.region.back')}</kbd> back
      {:else}
        per-class letter assigns · <kbd>{kg('review.queue.class_picker')}</kbd> search all
        classes ·
        <kbd>{kg('review.queue.confirm')}</kbd> confirm · <kbd>{kg('review.skip')}</kbd>
        skip · <kbd>{kg('review.queue.discard')}</kbd> discard ·
        <kbd>{kg('review.undo')}</kbd> undo
      {/if}
    </span>
  </div>

  {#if unavailableTabNotice}
    <div
      class="mx-4 mt-3 flex items-center gap-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200"
      data-testid="tab-unavailable"
      role="status"
    >
      <span class="grow">{unavailableTabNotice}</span>
      <button type="button" class="btn-sm" onclick={() => (unavailableTabNotice = null)}
        >Dismiss</button
      >
    </div>
  {/if}

  <!-- Body -->
  <!-- F8 D4: below lg the two panels stack; the body scrolls as a whole
       there instead of squeezing each panel into half the height (the
       metadata pane was ~79px tall at 800px). -->
  <div
    class="grid min-h-0 flex-1 grid-cols-1 content-start gap-4 overflow-y-auto p-4 lg:grid-cols-2 lg:content-normal lg:overflow-hidden"
    data-testid="review-body"
  >
    {#if queue.loading && queue.items.length === 0}
      <p class="col-span-full text-sm text-zinc-500">Loading...</p>
    {:else if awaitingDeepLink}
      <!-- DQ-M7: page 1 (item #1) may already be loaded underneath this —
           don't render it, or the operator briefly sees and could act on
           the wrong crop while the target page is still being located. -->
      <p class="col-span-full text-sm text-zinc-500">Locating crop…</p>
    {:else if queue.error}
      <p class="col-span-full text-sm text-red-300">API unavailable: {queue.error}</p>
    {:else if !current}
      <!-- R3 (visual audit 2026-09-24): say WHY the queue is empty, from
           the served tab description and sort-fallback reason, instead of
           a bare "Queue empty.". -->
      <div class="col-span-full max-w-2xl text-sm" data-testid="queue-empty">
        <p class="text-zinc-300">{emptyMessage.title}</p>
        {#each emptyMessage.lines as line (line)}
          <p class="mt-1 text-xs text-zinc-500">{line}</p>
        {/each}
        {#if emptyMessage.link}
          <!-- #36 item 9: served empty_state says the prerequisite this
               queue needs has never been computed — point straight at the
               control, not just "empty". -->
          <p class="mt-2 text-xs">
            <a
              class="text-blue-400 underline hover:text-blue-300"
              href={resolve(projectHref(emptyMessage.link.href))}
              >{emptyMessage.link.text}</a
            >
          </p>
        {/if}
        <!-- The served empty_state offers the exact request that embeds the
             items this queue skips; renders only when it applies. -->
        <p class="mt-2 text-xs">
          <EmptyQueueEmbed
            emptyState={reviewTabsVocabularyStore.emptyState}
            reasonText={`${emptyReason ?? ''} ${sortFallbackReason ?? ''}`}
          />
        </p>
      </div>
    {:else}
      <!-- Source image with bbox -->
      <div
        class="flex h-[35vh] flex-col surface p-2 lg:h-auto lg:min-h-0"
        data-testid="review-source-panel"
      >
        <div class="mb-2 flex items-center gap-2 px-1 text-xs text-zinc-400">
          <span>source</span>
          <span class="grow"></span>
          <span class="font-mono">{current.source ?? ''}</span>
        </div>
        <!-- V-3: top-aligned, so on a tall pane the image sits at the top
             rather than mid-way down an empty black panel. -->
        <div class="flex min-h-0 flex-1 items-start justify-center bg-zinc-950">
          <!-- K6: boxes/labels are drawn client-side from
               GET {API_PREFIX}/crops/{id}/context — the server no longer
               burns an overlay into this image. -->
          <SourceImageOverlay cropId={current.id} maxDim={1280} align="start" />
        </div>
      </div>

      <!-- Crop + meta -->
      <div class="flex flex-col surface p-2 lg:min-h-0" data-testid="review-crop-panel">
        <div class="mb-2 flex items-center gap-2 px-1 text-xs text-zinc-400">
          <span>crop</span>
          <span class="grow"></span>
          <span class="font-mono">{current.id.slice(0, 12)}…</span>
        </div>
        <!-- M2/M12 (2026-09-24 interactive pass): shrink-0 + a floor
             height so this panel never collapses toward 0px when the
             content below it (Details, slot fields) grows past the
             column's fixed height — that content scrolls in its own
             region instead of squeezing the image.
             DQ-M5: `max-h-[40%]` is the other half of the fix — a ceiling
             on top of that floor, so a tall/tiny crop's `h-full` fill (the
             phase-A p9 "upscale to fit" behavior) can no longer consume
             the whole panel and push Reason/Proposed/Confirm-Skip-Discard
             off-screen. The floor itself came down from M2/M12's original
             300px to 210px in the same pass: at a real 1280×720 viewport
             the 300px floor alone (before any tall crop even enters the
             picture) already pushed the action buttons 3px past the
             bottom edge — e2e/stubbed/test_review_crop_viewport.py
             pins this exact budget. 210px keeps a region sub-box legible
             (the box canvas is still square-aspect within it) while fitting.
             Verified at 1280×720, 1600×1000 and 1920×1080 — see
             artifacts_local/cw-live/phase-b-fixes/. -->
        <!-- F8 D4: overflow-hidden so a region canvas never paints over the
             first metadata row (the Reason row read half-clipped). -->
        <div
          class="flex h-[210px] shrink-0 items-center justify-center overflow-hidden bg-zinc-950 lg:h-auto lg:max-h-[40%] lg:min-h-[210px]"
        >
          {#if isMultiBoxSlot && activeSlot}
            <!-- W8 multi-box (docs/design/w8-multibox-frontend-plan-2026-09-26.md):
                 every box drawn at once, numbered by list position. Same
                 select/add/delete/Tab interaction in scan and edit mode;
                 only edit mode allows geometry changes (readonly canvas
                 in scan). -->
            <MultiBoxCanvas
              bind:this={slotCanvas}
              cropId={current.id}
              boxes={multiBox.boxes
                .filter((b) => b.box != null)
                .map((b) => ({
                  box: b.box!,
                  state: b.state,
                  label: `${activeSlot!.label.title} (${multiBoxStateLabel(b.state)})`,
                  locked: b.boxId != null && lockedBoxIds.has(b.boxId),
                }))}
              selectedIndex={multiBox.selectedIndex}
              busy={multiBox.busy}
              readonly={!editMode}
              maxBoxes={multiBox.maxBoxes}
              ringColorFor={multiBoxRingColor}
              dashedFor={multiBoxDashed}
              onselect={(i) => multiBox.select(i)}
              onnext={() => multiBox.next()}
              onmove={(_i, box) => multiBox.moveSelected(box)}
              onadd={(box) => multiBox.addBox(box)}
              ondelete={() => multiBox.deleteSelected()}
              class="aspect-square w-auto h-full max-h-full min-w-0 max-w-full"
            />
          {:else}
            <!-- p9 (2026-09-24 interactive pass): `max-h-full max-w-full`
                 only ever shrinks — a thumbnail smaller than its
                 container (common at 1920, where this panel stretches
                 to ~900px tall but the served thumb is a few hundred px)
                 rendered at its tiny natural size instead of upscaling
                 to fill the space. `h-full w-full` + object-contain fills
                 the container either direction, still preserving aspect
                 ratio.
                 DQ-M5: unbounded, that fill upscaled a 98×106 crop to
                 598-918px tall depending on viewport, pushing the actions
                 below it off-screen — this is a regression from p9, which
                 fit width but not height. The container above is now
                 height-capped (`max-h-[40%]`); `cropDisplayStyle` (inline,
                 from capCropDisplayStyle()) is the second half — it bounds
                 the *rendered* size to at most CROP_UPSCALE_CAP× the
                 crop's own natural pixels, read back via
                 bind:naturalWidth/naturalHeight, so a tiny crop no longer
                 blows up to fill whatever room the container has even
                 when that room is generous. Empty until the image has
                 loaded (natural size unknown), during which `h-full
                 w-full` still applies as the pre-p9 fallback. -->
            <img
              src={getThumbUrl(current.id, 384)}
              alt="crop"
              loading="lazy"
              decoding="async"
              bind:naturalWidth={cropNaturalWidth}
              bind:naturalHeight={cropNaturalHeight}
              style={cropDisplayStyle}
              class="h-full w-full object-contain"
            />
          {/if}
        </div>

        {#if isMultiBoxSlot && activeSlot}
          <!-- One state chip per box, click to select (mirrors
               MultiBoxCanvas's numbered rings). The served
               region_set_complete === false means the VLM reported
               visible regions that are missing from the list. -->
          <div class="mt-2 flex flex-wrap gap-1" data-testid="multibox-chips">
            {#each multiBox.boxes as b, i (b.boxId ?? `new-${i}`)}
              <button
                type="button"
                class="chip {i === multiBox.selectedIndex
                  ? 'border-sky-500/60 bg-sky-500/15 text-sky-100'
                  : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800'}"
                onclick={() => multiBox.select(i)}
              >
                #{i + 1}
                {multiBoxStateLabel(b.state)}
              </button>
            {/each}
            {#if multiBox.boxes.length === 0}
              <span class="text-[11px] text-zinc-500">no boxes</span>
            {/if}
            {#if current && slotOf(current, activeSlot)?.boxSet?.setComplete === false}
              <span
                class="chip border-amber-500/40 bg-amber-500/15 text-amber-200"
                title="The VLM reported visible regions that are missing from this list"
                data-testid="set-incomplete"
              >
                set incomplete
              </span>
            {/if}
            {#if multiBox.maxBoxes != null}
              <!-- W8.8: the served region_profile.limits.max_boxes_per_write —
                   never a client-guessed cap. -->
              <span
                class="text-[11px] text-zinc-500"
                title="Served limit on boxes per write (region_profile.limits.max_boxes_per_write)"
              >
                {multiBox.boxes.length} / {multiBox.maxBoxes} max
              </span>
            {/if}
          </div>
          {#if current}
            <VectorRefreshNotice cropId={current.id} refresh={multiBox.vectorRefresh} />
          {/if}
        {/if}

        <!-- Everything below the image scrolls in its own region — the
             image above keeps its floor height regardless of how much
             metadata/Details content is open. -->
        <div
          class="mt-3 pr-1 lg:min-h-0 lg:flex-1 lg:overflow-y-auto"
          data-testid="review-meta-pane"
        >
          <dl class="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
            {#if currentSlotRejectionReason}
              {@const reasonKind = regionVocabularyStore.rejectionReasonKind(
                currentSlotRejectionReason,
              )}
              <dt class="text-zinc-500">
                {reasonKind === 'needs_human' ? 'Needs review' : 'Rejection'}
              </dt>
              <dd
                class={reasonKind === 'model_verdict'
                  ? 'text-red-300'
                  : reasonKind === 'needs_human'
                    ? 'text-zinc-200'
                    : 'text-amber-300'}
              >
                {regionVocabularyStore.rejectionReasonLabel(currentSlotRejectionReason)}
              </dd>
            {:else}
              <dt class="text-zinc-500">Reason</dt>
              <dd class="text-zinc-200">{current.reason ?? '—'}</dd>
            {/if}

            <dt class="text-zinc-500">Current label</dt>
            <dd class="text-zinc-200">
              <!-- R6 (visual audit 2026-09-24): a class-less item used to
                   read "Current label (vlm)" with a blank value. -->
              {#if current.class_name}
                {current.class_name}
              {:else}
                <span class="italic text-zinc-500">{NO_CLASS_YET}</span>
              {/if}
              {#if current.label_source}
                <span class="ml-1 text-zinc-500">({current.label_source})</span>
              {/if}
              <!-- dq-queues cutover (2026-09-24): vlm_raw_class/
                   vlm_class_empty_reason fold into this row rather than
                   their own — the review panel is height-budgeted
                   (DQ-M5, max-h-[46%] crop cap so the action buttons
                   stay visible without scrolling at 1280x720), and every
                   extra dt/dd row eats into that budget. -->
              {#if current.vlm_raw_class}
                <span class="ml-1 text-[11px] text-zinc-500"
                  >— VLM said: {current.vlm_raw_class}</span
                >
              {:else if current.vlm_class_empty_reason}
                <span class="ml-1 text-[11px] text-orange-300"
                  >— {vlmEmptyReasonText(current.vlm_class_empty_reason)}</span
                >
              {/if}
            </dd>

            <!-- m1 (2026-09-24 interactive pass): a name-only proposal
                 (proposed_class_id absent — proposed_class_name a
                 non-registry term) used to render in
                 the same confirmable
                 yellow style as a real proposal, though Enter opens the
                 picker instead of confirming it — the row now says so. -->
            <dt class="text-zinc-500">Proposed</dt>
            {#if current.proposed_class_name && current.proposed_class_id == null}
              <dd
                class="text-zinc-400"
                title="Not a registry class — Enter opens the picker"
              >
                {current.proposed_class_name} <span class="text-[10px]">(hint only)</span>
              </dd>
            {:else}
              <dd class="text-yellow-200">{current.proposed_class_name ?? '—'}</dd>
            {/if}

            {#if opinion.kind === 'no_opinion'}
              <!-- F8 D1: out of the probe's classes: no opinion, never
                   shown as a prediction or as agreement. -->
              <dt class="text-zinc-500">Model predicts</dt>
              <dd class="text-zinc-400" data-testid="probe-no-opinion">
                {NO_OPINION_TEXT}
              </dd>
            {:else if opinion.kind === 'unsure'}
              <!-- OpenProcessor 8990ede: a served disagreement below the
                   server's own confidence threshold (`probe_actionable`
                   false) — shown, but never offered as an Accept action. -->
              <dt class="text-zinc-500">Model predicts</dt>
              <dd class="text-zinc-400" data-testid="probe-unsure">
                model unsure: {current.probe_pred_class}
              </dd>
            {:else if opinion.kind === 'prediction'}
              <!-- G4 closed 2026-09-24 (logic-moves item 14): the backend
                 now serves `probe_pred_class_id` alongside the display
                 name, so "Accept" no longer needs a client-side
                 name→id lookup — assign() takes the served id directly. -->
              <dt class="text-zinc-500">Model predicts</dt>
              <dd class="flex flex-wrap items-center gap-1.5 text-zinc-200">
                {current.probe_pred_class}
                {#if current.probe_pred_entropy != null}
                  <ScoreChip
                    label="entropy"
                    value={current.probe_pred_entropy}
                    size="sm"
                  />
                {/if}
                {#if opinion.showAccept}
                  <button
                    type="button"
                    class="rounded border border-blue-500/60 bg-blue-500/15 px-1.5 py-0.5 text-[11px] text-blue-100 hover:bg-blue-500/25"
                    onclick={acceptModelClass}
                  >
                    Accept model's class
                  </button>
                {/if}
              </dd>
            {/if}

            {#if current.needs_new_class}
              <dt class="text-zinc-500">Needs new class</dt>
              <dd class="text-amber-200">
                {current.needs_new_class_note ||
                  'flagged — no matching class in the registry'}
              </dd>
            {/if}

            <!-- DQ-M8 (docs/design/data-quality-pass-2026-09-24.md):
                 `label_confidence` (wire `confidence`) is the classifier-
                 detector score on every row, including VLM-sourced
                 ones — the repro was exactly this panel, "Current label
                 widget_a (vlm)" directly above "Confidence 94.6%", which
                 reads as the VLM's own certainty. Label it for what it is
                 whenever the current label came from the VLM (served
                 role, not a hardcoded string match), and show the VLM's
                 own categorical confidence (`vlm_confidence`, served
                 separately) as its own row when there's a real number to
                 contrast it with. -->
            <dt class="text-zinc-500">
              {isCurrentLabelVlmSourced ? 'Detector score' : 'Confidence'}
            </dt>
            <dd class="font-mono">
              {current.label_confidence != null
                ? `${(current.label_confidence * 100).toFixed(1)}%`
                : '—'}
              <!-- dq-queues cutover (2026-09-24): class_confidence (the
                   confidence of whoever set the CLASS label — distinct
                   from label_confidence above, always the detector
                   score) folds in here rather than its own row, only
                   when there's no VLM-confidence row below to carry it
                   instead — see that row's comment for why (DQ-M5
                   height budget). -->
              {#if current.class_confidence != null && !current.vlm_confidence}
                <span class="ml-1 text-[11px] text-zinc-500">
                  · label: {current.class_confidence_source === 'model' ? 'Model ' : ''}{(
                    current.class_confidence * 100
                  ).toFixed(0)}%
                </span>
              {/if}
            </dd>

            {#if current.vlm_confidence}
              <dt class="text-zinc-500">VLM confidence</dt>
              <dd class="font-mono text-zinc-200">
                {current.vlm_confidence}
                <!-- class_confidence is the numeric mapping of this same
                     categorical value when VLM-sourced — shown inline
                     rather than its own row (DQ-M5 height budget, see
                     the Detector score/Confidence row's comment above). -->
                {#if current.class_confidence != null}
                  <span class="text-[11px] text-zinc-500"
                    >({(current.class_confidence * 100).toFixed(0)}%)</span
                  >
                {/if}
              </dd>
            {/if}

            {#if current.proposal_name}
              <dt class="text-zinc-500">Proposal hint</dt>
              <dd>
                <span
                  class="rounded border border-cyan-500/40 bg-cyan-500/15 px-1.5 py-0.5 text-[11px] text-cyan-200"
                  title="A general-purpose detector found a subject here that the primary classifier missed. Coarse class — pick the specific class below (a close variant may be near one-click)."
                >
                  {current.proposal_name}
                </span>
              </dd>
            {/if}

            {#if current.crop_rank_in_image != null || current.blur_lap_ratio != null}
              <dt class="text-zinc-500">Rank · clarity</dt>
              <dd class="font-mono text-zinc-300">
                {current.crop_rank_in_image != null
                  ? current.crop_rank_in_image === 1
                    ? '★1 largest'
                    : `#${current.crop_rank_in_image}`
                  : '—'}
                {#if current.blur_lap_ratio != null}
                  · b{current.blur_lap_ratio.toFixed(2)}
                {/if}
              </dd>
            {/if}

            {#if current.mistakenness_score != null || (current && searchScores.has(current.id))}
              <dt class="text-zinc-500">Scores</dt>
              <dd class="flex flex-wrap items-center gap-1.5">
                {#if current.mistakenness_score != null}
                  <ScoreChip
                    label="mistakenness"
                    value={current.mistakenness_score}
                    method={current.mistakenness_method}
                    version={current.mistakenness_version}
                    size="sm"
                  />
                {/if}
                {#if current && searchScores.has(current.id)}
                  <ScoreChip
                    label="match"
                    value={searchScores.get(current.id) ?? 0}
                    size="sm"
                  />
                {/if}
              </dd>
            {/if}
          </dl>

          {#if activeSlot && slotLabels}
            {@const slotData = slotOf(current, activeSlot)}
            <!-- Slot inline review. The canvas above is live — drag/resize
               the proposal in place and hit Enter to confirm. The Reject
               button (or D) marks the slot's rejectState. The whole flow
               is two keystrokes per crop on average: minor twitch with
               arrows / handles, then Enter. -->
            <div class="mt-3 grid grid-cols-2 gap-y-1 text-xs">
              <span class="text-zinc-500">{slotLabels.scoreLabel}</span>
              <span class="font-mono text-zinc-200">
                {#if isMultiBoxSlot && selectedSlotBox?.score != null}
                  {(selectedSlotBox.score * 100).toFixed(1)}%
                {:else if !isMultiBoxSlot && slotData?.subBox?.score != null}
                  {(slotData.subBox.score * 100).toFixed(1)}%
                {:else}
                  —
                {/if}
              </span>
              <span class="text-zinc-500">{slotLabels.statusLabel}</span>
              <span class="flex items-center gap-1.5">
                <select
                  bind:value={editedSlotStatus}
                  onchange={() => void commitSlotStatus()}
                  class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-xs text-zinc-100 focus:border-blue-500 focus:outline-none"
                >
                  <option value="">—</option>
                  {#each slotStatusOptions as opt (opt.value)}
                    <option value={opt.value}>{opt.label}</option>
                  {/each}
                </select>
                <!-- dq-region (2026-09-24): validated (human-only) vs
                     auto_confirmed (machine accepted, unreviewed) —
                     region_verified no longer distinguishes the two. -->
                {#if slotData?.lifecycle?.validated}
                  <span
                    class="rounded border border-emerald-500/40 bg-emerald-500/15 px-1 text-[10px] text-emerald-200"
                    title="A human confirmed/drew/rejected this region"
                  >
                    human validated
                  </span>
                {:else if slotData?.lifecycle?.autoConfirmed}
                  <span
                    class="rounded border border-blue-500/40 bg-blue-500/15 px-1 text-[10px] text-blue-200"
                    title="Auto-confirmed by the worker's policy — accepted but not yet reviewed by a human"
                  >
                    auto-confirmed
                  </span>
                {/if}
                <!-- The verifier's own box-correctness verdict for the
                     selected box: false is the "model said wrong box"
                     signal. -->
                {#if selectedSlotBox?.bboxCorrect === false}
                  <span
                    class="rounded border border-red-500/40 bg-red-500/15 px-1 text-[10px] text-red-200"
                    title="The verifier judged this box incorrect"
                  >
                    model: box wrong
                  </span>
                {/if}
                {#if selectedSlotBox?.locked}
                  <span
                    class="rounded border border-zinc-600 bg-zinc-800 px-1 text-[10px] text-zinc-300"
                    title="A human created or edited this box; automated stages leave it alone"
                  >
                    locked
                  </span>
                {/if}
                {#if slotData?.boxSet?.setComplete === false}
                  <span
                    class="rounded border border-amber-500/40 bg-amber-500/15 px-1 text-[10px] text-amber-200"
                    title="The VLM reported visible regions that are missing from this list"
                  >
                    set incomplete
                  </span>
                {/if}
              </span>
              <span class="text-zinc-500">Detector</span>
              <span class="flex flex-wrap items-center gap-1.5">
                {#if selectedSlotBox?.detector || slotData?.provenance?.detector}
                  <ProvenanceChip
                    detector={selectedSlotBox?.detector ??
                      slotData?.provenance?.detector ??
                      null}
                    version={selectedSlotBox?.detectorVersion ??
                      slotData?.provenance?.detectorVersion ??
                      null}
                  />
                {:else}
                  <span class="text-zinc-500">—</span>
                {/if}
                {#if slotData?.provenance?.verifier}
                  <ProvenanceChip
                    detector={slotData.provenance.verifier}
                    tag="verify"
                    version={slotData.provenance.verifierVersion}
                    size="sm"
                  />
                {/if}
                <!-- The mistakenness chip lives in the Scores row above, not
                     next to the detector provenance chips. -->
                {#if isMultiBoxSlot && activeSlot}
                  <!-- W8: no separate "candidate" concept — a
                       verifier-rejected box is just a SlotBox with
                       state: 'rejected' and its own rejectionReason
                       (owner decision 2026-09-26, no backward
                       compatibility). Show a needs-review/rejected chip
                       per rejected box, styled by the served kind, same
                       as before — worded per-box, not per-item. -->
                  <RejectedBoxChips
                    boxes={multiBox.boxes}
                    subBoxes={slotData?.subBoxes}
                  />
                  {#if multiBox.boxes.length === 0 && !editMode}
                    <span
                      class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-[10px] text-zinc-400"
                    >
                      no boxes · press {kg('review.region.edit_box')} to draw
                    </span>
                  {/if}
                {/if}
              </span>
              {#if slotData?.provenance?.chain && slotData.provenance.chain.length > 0}
                <span class="text-zinc-500">Cascade</span>
                <span class="flex flex-wrap items-center gap-1">
                  {#each slotData.provenance.chain as entry, i (i)}
                    <ProvenanceChip raw={entry} size="sm" />
                  {/each}
                </span>
              {/if}
              <!-- OpenProcessor W1 (text-free regions): a slot with no
                   text capability renders no text row/edit at all — not
                   even a disabled input — since the backend 422s
                   `region_text` on such a profile. -->
              {#if activeSlot.capabilities.text}
                <span class="text-zinc-500">{slotLabels.textLabel}</span>
                <span class="flex items-center gap-1.5">
                  <input
                    type="text"
                    bind:value={editedSlotText}
                    onblur={() => void commitSlotText()}
                    onkeydown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        (e.currentTarget as HTMLInputElement).blur();
                      }
                    }}
                    placeholder={slotLabels.textPlaceholder}
                    disabled={isMultiBoxSlot && selectedSlotBox == null}
                    spellcheck="false"
                    autocapitalize={activeSlot.capabilities.text?.transform ===
                    'uppercase'
                      ? 'characters'
                      : 'off'}
                    class="w-28 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-xs text-zinc-100 placeholder:italic placeholder:text-zinc-600 focus:border-blue-500 focus:outline-none {activeSlot
                      .capabilities.text?.monospace
                      ? 'font-mono'
                      : ''}"
                  />
                  {#if isMultiBoxSlot ? selectedSlotBox?.textSource : slotData?.text?.source}
                    <ProvenanceChip
                      detector={(isMultiBoxSlot
                        ? selectedSlotBox?.textSource
                        : slotData?.text?.source) ?? null}
                      size="sm"
                    />
                  {/if}
                  {#if (isMultiBoxSlot ? selectedSlotBox?.textConfidence : slotData?.text?.confidence) != null}
                    <span class="text-[10px] text-zinc-500">
                      {(
                        ((isMultiBoxSlot
                          ? selectedSlotBox?.textConfidence
                          : slotData?.text?.confidence) ?? 0) * 100
                      ).toFixed(0)}%
                    </span>
                  {/if}
                  {#if selectedSlotBox?.textDisagreement}
                    <span
                      class="rounded border border-orange-500/40 bg-orange-500/15 px-1 text-[10px] text-orange-200"
                      title="vlm: {selectedSlotBox.textVlm ??
                        '∅'} · ocr: {selectedSlotBox.textOcr ?? '∅'}"
                    >
                      readers disagree
                    </span>
                  {/if}
                  <!-- Why the chosen reading won / why the VLM's own
                       reading was rejected as not text (served
                       vocabulary ids). -->
                  {#if selectedSlotBox?.textChoice && selectedSlotBox.textChoice !== 'human'}
                    <span
                      class="rounded border border-zinc-700 bg-zinc-900 px-1 text-[10px] text-zinc-400"
                      title="How this reading was chosen"
                    >
                      {regionVocabularyStore.textChoiceLabel(selectedSlotBox.textChoice)}
                    </span>
                  {/if}
                  {#if selectedSlotBox?.textVlmInvalid}
                    <span
                      class="rounded border border-red-500/40 bg-red-500/15 px-1 text-[10px] text-red-200"
                      title="Why the VLM's own reading wasn't used as text"
                    >
                      vlm invalid: {regionVocabularyStore.invalidReasonLabel(
                        selectedSlotBox.textVlmInvalid,
                      )}
                    </span>
                  {/if}
                </span>
              {/if}
              {#if statusWantsRejectionReason(activeSlot, editedSlotStatus, regionStatusesStore.list)}
                {@const servedReasonKind = regionVocabularyStore.rejectionReasonKind(
                  slotData?.lifecycle?.rejectionReason,
                )}
                <span class="text-zinc-500">Rejection reason</span>
                <span>
                  {#if servedReasonKind && editedRejectionReason === (slotData?.lifecycle?.rejectionReason ?? '')}
                    <!-- R6 (visual audit 2026-09-24): a machine reason from the
                         served vocabulary (e.g. verifier_no_verdict) used to
                         sit raw in an editable box; show its served label
                         instead. Changing the status still re-asks. -->
                    <span class="text-zinc-300" data-testid="served-rejection-reason"
                      >{regionVocabularyStore.rejectionReasonLabel(
                        slotData?.lifecycle?.rejectionReason,
                      )}</span
                    >
                  {:else}
                    <input
                      type="text"
                      bind:value={editedRejectionReason}
                      onblur={() => void commitRejectionReason()}
                      onkeydown={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault();
                          (e.currentTarget as HTMLInputElement).blur();
                        }
                      }}
                      placeholder="e.g. blurred, occluded, glare"
                      class="w-44 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-xs text-zinc-100 focus:border-blue-500 focus:outline-none"
                    />
                  {/if}
                </span>
              {/if}
            </div>
            <div
              class="sticky bottom-0 z-10 mt-3 flex flex-wrap gap-2 bg-[rgb(var(--bg-elevated))] py-1"
              data-testid="review-actions"
            >
              {#if editMode}
                <button
                  class="btn btn-primary"
                  type="button"
                  onclick={confirmMultiBoxSlot}
                  disabled={multiBox.busy}
                >
                  Save boxes
                </button>
                <button
                  class="btn"
                  type="button"
                  onclick={toggleEdit}
                  disabled={multiBox.busy}
                >
                  Cancel
                </button>
              {:else}
                <button
                  class="btn btn-primary"
                  type="button"
                  onclick={isMultiBoxSlot ? confirmMultiBoxSlot : confirmSlot}
                >
                  {slotLabels.confirmLabel}
                </button>
                <button class="btn btn-danger" type="button" onclick={rejectSlot}>
                  {slotLabels.rejectLabel}
                </button>
                {#if activeSlot.capabilities.lifecycle?.falsePositiveState}
                  <button
                    class="btn"
                    type="button"
                    onclick={markFalsePositive}
                    title="Detector drew a box but it's not the {activeSlot.label
                      .singular} — keep the box as a training hard negative ({kg(
                      'review.region.false_positive',
                    )})"
                  >
                    False positive
                  </button>
                {/if}
                <button class="btn" type="button" onclick={skip}>Skip</button>
                {#if isMultiBoxSlot}
                  <button
                    class="btn"
                    type="button"
                    onclick={toggleEdit}
                    aria-pressed={editMode}
                    title="Toggle bbox edit mode ({kg('review.region.edit_box')})"
                  >
                    Edit boxes
                  </button>
                {/if}
                <button
                  class="btn"
                  type="button"
                  onclick={slotBack}
                  disabled={slotUndoStack.length === 0}
                  title="Re-open the most-recently confirmed {activeSlot.label
                    .singular} ({kg(
                    'review.region.back',
                  )}) — only re-queues it locally, use {kg(
                    'review.undo',
                  )} to undo the server write"
                >
                  ← Step back
                </button>
              {/if}
            </div>
            {#if slotUndoStack.length > 0}
              <!-- DQ-m6 (docs/design/data-quality-pass-2026-09-24.md):
                   this stack holds confirm, reject AND false-positive
                   entries — the
                   footer said "confirmed" unconditionally, so a reject
                   read as "1 confirmed in this session". "Actioned" is
                   accurate for all three. -->
              <p class="mt-1 text-[10px] text-zinc-500">
                {slotUndoStack.length} actioned in this session — press {kg(
                  'review.region.back',
                )} to step back.
              </p>
            {/if}
          {:else}
            <div
              class="sticky bottom-0 z-10 mt-3 flex flex-wrap gap-2 bg-[rgb(var(--bg-elevated))] py-1"
              data-testid="review-actions"
            >
              <button
                class="btn btn-primary"
                type="button"
                onclick={confirmAndAdvance}
                disabled={!canConfirm}
                title={canConfirm
                  ? undefined
                  : `No proposed class on this item — press ${kg('review.queue.class_picker')} or ${kg('review.queue.confirm')} to search.`}
              >
                Confirm
              </button>
              <button class="btn" type="button" onclick={skip}>Skip</button>
              <button class="btn btn-danger" type="button" onclick={discard}
                >Discard</button
              >
              <button class="btn" type="button" onclick={undoLast}>Undo</button>
            </div>
          {/if}

          <!-- Most-validated classes — click to label OR press the per-class
             hotkey configured on /classes. Hotkey badges only show for
             classes the user has explicitly bound (otherwise the strip is
             still clickable, just no kbd hint). The class strip is hidden
             on the regions tab; class assignment isn't relevant there. -->
          {#if !isSlotTab(tab)}
            <div class="mt-3 flex flex-wrap gap-1.5">
              {#each topClasses as cls (cls.id)}
                <button
                  type="button"
                  class="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200
                     hover:border-blue-500/60 hover:bg-blue-500/10 hover:text-white
                     focus:outline-none focus:ring-2 focus:ring-blue-500/40"
                  title={cls.hotkey_letter
                    ? `Assign ${cls.name} (press ${cls.hotkey_letter})`
                    : `Assign ${cls.name}`}
                  onclick={() => assign(cls.id)}
                >
                  {#if cls.hotkey_letter}
                    <kbd
                      class="mr-1.5 rounded bg-zinc-800 px-1 py-0.5 font-mono text-[10px] uppercase text-blue-300"
                    >
                      {cls.hotkey_letter}
                    </kbd>
                  {/if}
                  {cls.name}
                </button>
              {/each}
              <!-- P1-4: only the 10 most-validated classes are one click above;
                 this opens the fuzzy-search picker over all non-deprecated
                 classes (same action as pressing /). -->
              <button
                type="button"
                class="rounded border border-dashed border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-400
                   hover:border-blue-500/60 hover:bg-blue-500/10 hover:text-white
                   focus:outline-none focus:ring-2 focus:ring-blue-500/40"
                title="Search all classes ({kg('review.queue.class_picker')})"
                onclick={openPicker}
              >
                <kbd
                  class="mr-1.5 rounded bg-zinc-800 px-1 py-0.5 font-mono text-[10px] text-blue-300"
                >
                  {kg('review.queue.class_picker')}
                </kbd>
                search all classes…
              </button>
            </div>
            <p class="mt-1.5 text-[10px] text-zinc-500">
              Click a class, press its bound letter, or press {kg(
                'review.queue.class_picker',
              )} to search all classes (set hotkeys on /classes).
            </p>
          {/if}

          <!-- G7/G9/G8: history + source image/siblings + item-text lines,
             via the same CropMetaPanel used by the /clusters detail
             modal — collapsed by default so it doesn't compete with the
             confirm/reject flow above. -->
          <div class="mt-3 border-t border-zinc-800 pt-2">
            <button
              type="button"
              class="text-[10px] uppercase tracking-wider text-zinc-500 hover:text-zinc-300"
              onclick={() => (detailsOpen = !detailsOpen)}
            >
              {detailsOpen ? '▾' : '▸'} Details
            </button>
            {#if detailsOpen}
              <div class="mt-2">
                <CropMetaPanel
                  crop={current}
                  embedded
                  onreprocessed={(c) => {
                    const idx = queue.items.findIndex((x) => x.id === c.id);
                    if (idx >= 0)
                      queue.items[idx] = { ...queue.items[idx], ...c } as ReviewItem;
                  }}
                />
              </div>
            {/if}
          </div>
        </div>
      </div>
    {/if}
  </div>

  <!-- Status bar — review is one-at-a-time so there's no "scroll to load more"
       affordance; the queue auto-fetches the next page in the background as
       the cursor advances toward the end of the loaded items. -->
  <div
    class="flex items-center justify-between gap-3 border-t border-zinc-800 px-4 py-2 text-sm"
  >
    <span class="font-mono text-xs text-zinc-500">
      {queue.items.length > 0 ? currentPosition : 0} / {queue.total}
      {#if queue.items.length < queue.total}
        <span class="ml-1 text-zinc-600">(loaded {queue.items.length})</span>
      {/if}
    </span>
    <span class="font-mono text-xs text-zinc-400">
      {#if queue.loadingMore}loading more…{:else if !queue.hasMore && queue.items.length > 0}all
        loaded{:else if queue.hasMore}auto-fetching{/if}
    </span>
  </div>
</div>

{#if dismissedPanelOpen}
  <!-- Minimal un-dismiss surface: crops discard() has permanently
       dismissed from every review queue, each restorable via
       reviewUndismissCrop. Backdrop click and Esc both close. -->
  <div
    class="fixed inset-0 z-50 flex items-start justify-center bg-black/50 pt-24"
    onclick={() => (dismissedPanelOpen = false)}
    onkeydown={(e) => e.key === 'Escape' && (dismissedPanelOpen = false)}
    role="presentation"
  >
    <div
      class="w-full max-w-lg overflow-hidden rounded-lg border border-zinc-700 bg-zinc-900 shadow-xl"
      onclick={(e) => e.stopPropagation()}
      role="presentation"
    >
      <div class="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
        <span class="text-sm font-medium text-zinc-200">Dismissed crops</span>
        <button
          type="button"
          class="text-xs text-zinc-500 hover:text-zinc-300"
          onclick={() => (dismissedPanelOpen = false)}
        >
          close
        </button>
      </div>
      <ul class="max-h-96 overflow-y-auto py-1 text-sm">
        {#if dismissedLoading}
          <li class="px-3 py-2 text-zinc-500">Loading…</li>
        {:else if dismissedError}
          <li class="px-3 py-2 text-red-300">
            Failed to load dismissed crops: {dismissedError}
          </li>
        {:else if dismissedItems.length === 0}
          <li class="px-3 py-2 text-zinc-500">No dismissed crops.</li>
        {/if}
        {#each dismissedItems as crop (crop.id)}
          <li class="flex items-center justify-between gap-2 px-3 py-1.5">
            <span class="truncate text-zinc-300">{crop.class_name ?? crop.id}</span>
            <button
              type="button"
              class="shrink-0 rounded border border-zinc-700 px-2 py-0.5 text-xs text-zinc-300 hover:border-blue-500 hover:text-blue-300"
              onclick={() => undismiss(crop)}
            >
              Restore
            </button>
          </li>
        {/each}
      </ul>
    </div>
  </div>
{/if}

{#if rejectReasonPromptOpen}
  <!-- DQ-m6: in-app replacement for the old window.prompt() — see
       promptForRejectionReason()'s doc comment for why. Backdrop click
       and Esc both cancel (reason stays null, matching the old
       "Cancel" prompt() behavior); Enter submits. -->
  <div
    class="fixed inset-0 z-50 flex items-start justify-center bg-black/50 pt-24"
    onclick={cancelRejectReasonPrompt}
    role="presentation"
  >
    <div
      class="w-full max-w-sm overflow-hidden rounded-lg border border-zinc-700 bg-zinc-900 p-3 shadow-xl"
      onclick={(e) => e.stopPropagation()}
      role="presentation"
      use:trapFocus={{ onEscape: cancelRejectReasonPrompt }}
    >
      <label class="mb-2 block text-sm text-zinc-300" for="reject-reason-input">
        Reason for rejecting this {activeSlot?.label.singular ?? 'item'} (optional):
      </label>
      <input
        id="reject-reason-input"
        type="text"
        use:focusOnMount
        bind:value={rejectReasonPromptValue}
        onkeydown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            submitRejectReasonPrompt();
          } else if (e.key === 'Escape') {
            e.preventDefault();
            cancelRejectReasonPrompt();
          }
        }}
        class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
        placeholder={`e.g. blurry, wrong angle, not a ${activeSlot?.label.singular ?? 'match'}…`}
      />
      <div class="mt-3 flex justify-end gap-2">
        <button type="button" class="btn" onclick={cancelRejectReasonPrompt}>
          Cancel
        </button>
        <button type="button" class="btn btn-primary" onclick={submitRejectReasonPrompt}>
          Reject
        </button>
      </div>
    </div>
  </div>
{/if}

{#if pickerOpen}
  <!-- Class picker (P1-4) — fuzzy-search over every non-deprecated class,
       opened with / or the "search all classes…" button. Backdrop click
       and Esc both close without assigning. -->
  <div
    class="fixed inset-0 z-50 flex items-start justify-center bg-black/50 pt-24"
    onclick={closePicker}
    role="presentation"
  >
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div
      class="w-full max-w-md overflow-hidden rounded-lg border border-zinc-700 bg-zinc-900 shadow-xl"
      role="dialog"
      aria-modal="true"
      aria-label="Search classes"
      tabindex="-1"
      use:trapFocus={{ onEscape: closePicker }}
      onclick={(e) => e.stopPropagation()}
    >
      <input
        bind:this={pickerInputEl}
        bind:value={pickerQuery}
        type="text"
        placeholder="Search all {pickerClasses.length} classes…"
        class="w-full border-b border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:outline-none"
        onkeydown={onPickerKeydown}
      />
      <ul class="max-h-72 overflow-y-auto py-1 text-sm">
        {#if pickerResults.length === 0}
          <li class="px-3 py-2 text-zinc-500">No matching class.</li>
        {/if}
        {#each pickerResults as cls, i (cls.id)}
          <li>
            <button
              type="button"
              class="flex w-full items-center gap-2 px-3 py-1.5 text-left {i ===
              pickerIndex
                ? 'bg-blue-500/20 text-white'
                : 'text-zinc-200 hover:bg-zinc-800'}"
              onclick={() => pickClass(cls)}
              onmouseenter={() => (pickerIndex = i)}
            >
              {#if cls.hotkey_letter}
                <kbd
                  class="rounded bg-zinc-800 px-1 py-0.5 font-mono text-[10px] uppercase text-blue-300"
                >
                  {cls.hotkey_letter}
                </kbd>
              {/if}
              <span class="grow truncate">{cls.name}</span>
              <span class="shrink-0 text-[10px] text-zinc-500"
                >{cls.validated_count} validated</span
              >
            </button>
          </li>
        {/each}
      </ul>
      <p class="border-t border-zinc-800 px-3 py-1.5 text-[10px] text-zinc-500">
        <kbd>↑↓</kbd> navigate · <kbd>Enter</kbd> assign · <kbd>Esc</kbd> close
      </p>
    </div>
  </div>
{/if}
