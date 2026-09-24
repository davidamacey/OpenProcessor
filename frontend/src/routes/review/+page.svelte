<script lang="ts">
  import {
    cancelSelect,
    reviewDismissCrop,
    reviewUndismissCrop,
    getCrop,
    getCrops,
    getReviewQueue,
    getSelectStatus,
    getSourceImageWithBbox,
    getThumbUrl,
    locateInReviewQueue,
    putCropLabel,
    selectDiverse,
    setSlotBox,
    patchSlotMeta,
  } from '$lib/api';
  import BlurSlider from '$lib/components/BlurSlider.svelte';
  import ProvenanceChip from '$lib/components/ProvenanceChip.svelte';
  import BboxCanvas from '$lib/components/BboxCanvas.svelte';
  import ScoreChip from '$lib/components/ScoreChip.svelte';
  import ShortcutsButton from '$lib/components/ShortcutsButton.svelte';
  import SemanticSearchBox from '$lib/components/SemanticSearchBox.svelte';
  import StrategyBar from '$lib/components/StrategyBar.svelte';
  import SubjectScopeToggle from '$lib/components/SubjectScopeToggle.svelte';
  import { pushUndo, removeUndo, popUndo, reinsertAt } from '$lib/review/slotQueueOps';
  import { AbortRegistry } from '$lib/review/abortRegistry';
  import { buildSlotKeymap, rejectKeyGlyph } from '$lib/review/slotKeymap';
  import { isSlotSuppressedTab } from '$lib/review/slotTabGuard';
  import { computeViewBox } from '$lib/review/viewBox';
  import {
    humanWritableStates,
    statusClearsBox,
    statusWantsRejectionReason,
    panelLabels,
  } from '$lib/review/slotPanel';
  import { slotOf } from '$lib/annotations/cropSlots';
  import { resolveConfirmClassId, searchClasses } from '$lib/classPicker';
  import {
    endpointForTab,
    isSlotTab,
    REVIEW_PRESETS,
    REVIEW_TABS,
    resolveEffectiveTab,
    type ReviewPresetId,
    reviewDeepLink,
  } from '$lib/reviewTabs';
  import { isDiverseOverlayAvailable } from '$lib/strategies';
  import type {
    BBoxNorm,
    Crop,
    DiverseSelection,
    RegistryClass,
    ReviewItem,
    ReviewTab,
  } from '$lib/types';
  import { createPager } from '$lib/pager.svelte';
  import { createStrategyBar } from '$lib/strategyBar.svelte';
  import { isSemanticSearchAvailable } from '$lib/strategies';
  import { subscribeCurationEvents, type CurationEventSubscription } from '$lib/sse';
  import { untrack } from 'svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { undoStore } from '$stores/undo.svelte';
  import { regionStatusesStore } from '$stores/regionStatuses.svelte';
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { replaceState } from '$app/navigation';

  // Unified review by default — one continuous queue of every crop that
  // needs a human, sorted most-uncertain first. Down to 5 top-level tabs
  // (2026-09 consolidation, see $lib/reviewTabs.ts) — Mismatches / Gemma
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
  let pendingCropId: string | null = deepLink.cropId;
  // The slot backing the current tab, if any — the single derived value
  // P2.8b's mapping table (docs/genericization-plan-2026-09-13.md §9.5)
  // hangs every former `tab === 'plates'` call site off, instead of a
  // hand-maintained literal per site.
  const activeSlot = $derived(REVIEW_TABS.find((t) => t.id === tab)?.slot ?? null);
  // Active quick-filter preset chip on the All tab (null = plain All).
  // Only ever meaningful while tab === 'all' — resolveEffectiveTab drops
  // it for every other tab, and switching tabs clears it outright.
  let preset = $state<ReviewPresetId | null>(null);
  const effectiveTab = $derived<ReviewTab>(resolveEffectiveTab(tab, preset));
  function togglePreset(id: ReviewPresetId): void {
    preset = preset === id ? null : id;
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
  // Set from the review-queue response whenever the requested `?sort=`
  // couldn't be honored server-side (e.g. the field isn't backfilled
  // yet). Rendered as a small inline note, never a toast — this isn't a
  // failure, just a degraded request.
  let sortFallbackReason = $state<string | null>(null);
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
  const DIVERSE_K_MAX = 500;
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
    // term/terms filters (class_id, hdd_source) — NOT conf_min/conf_max/
    // min_blur_ratio/max_rank/plate text. Those controls are disabled in
    // the UI while diverseMode is active (see the filter bar below) so
    // this never silently drops something the operator thinks is applied.
    const f: Record<string, unknown> = {};
    if (sourceFilter) f.hdd_source = sourceFilter;
    if (classFilter != null) f.class_id = classFilter;
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
          // 'completed' moments later — a openprocessor timing quirk, out of
          // scope to fix here, but the frontend must still stop treating
          // this job as "ours" once it reports terminal).
          diverseError = st.error ?? `Diverse selection ${st.status}.`;
          diverseJobId = null;
          diverseJobStatus = null;
          resolve(null);
        } catch (e) {
          stopDiversePolling();
          diverseError = `Diverse selection failed: ${(e as Error).message}`;
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
      // (e.g. 'plates'), so this resolves through endpointForTab()
      // rather than forwarding the internal id directly.
      const res = await getReviewQueue(
        endpointForTab(effectiveTab) as ReviewTab,
        page,
        pageSize,
        _filter(),
      );
      sortFallbackReason = res.sort_fallback_reason ?? null;
      sortApplied = res.sort_applied ?? null;
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
  const handledIds = new Set<string>();

  // Filter bar. `sourceFilter` sends `source` to {API_PREFIX}/review/{tab} (item
  // 14/G3, 2026-09-24 logic-moves — renamed off the old `hdd_source`
  // control, which the endpoint never actually read). `termFilters()`
  // below (the diverse-selection scope, a different endpoint) still
  // sends the same value under `hdd_source` — that contract hasn't
  // changed.
  let sourceFilter = $state<string>('');
  let classFilter = $state<number | null>(null);
  let confMin = $state<number>(0);
  let confMax = $state<number>(1);
  // Plate-text search — only meaningful on tab=plates; ignored elsewhere
  // server-side. Surface in the filter strip when the operator is on
  // the plates tab.
  let plateTextQuery = $state<string>('');

  // Primary-subject controls (primary_low_conf / coco_blind_spots tabs).
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
    // GET /review/{tab} accepts class_id/source/conf_min/conf_max as of
    // the 2026-09-24 logic-moves cutover (item 14/G3 — verified live
    // against the real backend). Diverse mode disables these controls
    // (see the filter bar below) since POST {API_PREFIX}/select/diverse's
    // `scope.filters` doesn't support conf_min/conf_max at all, and
    // takes class_id/source through its own `termFilters()` instead of
    // this function.
    if (classFilter != null) f.class_id = classFilter;
    if (sourceFilter) f.source = sourceFilter;
    if (confMin > 0) f.conf_min = confMin;
    if (confMax < 1) f.conf_max = confMax;
    const textFilter = activeSlot?.capabilities.queue?.textFilter;
    if (textFilter && plateTextQuery) f[textFilter.param] = plateTextQuery;
    // max_rank / min_blur_ratio apply across every tab and preset — the
    // backend's own review.py comment says so explicitly ("Both apply
    // across tabs"). These used to be gated to only primary_low_conf /
    // coco_blind_spots, which meant the rank-scope and clarity controls
    // silently appeared/disappeared depending on which tab or quick-filter
    // chip was active — confusing and inconsistent with Conf/Class/Source,
    // which were never gated. Always available now, like those.
    if (subjectScope !== 0) f.max_rank = subjectScope;
    if (minBlurRatio != null) f.min_blur_ratio = minBlurRatio;
    Object.assign(f, strategyBar.toQueryParams());
    return f;
  }

  // Review is one-at-a-time — only `current` is rendered. We fetch one
  // page on tab/filter change and the cursor-arrow handler pulls the
  // next page as the user nears the end. The earlier eager prefetch
  // drained every page upfront, which on the busy 'all' tab fired ~4
  // chained network calls before first paint and made the page feel
  // frozen on slow connections. Lazy paging keeps first-paint snappy.
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
        toastStore.info(
          loc.reason
            ? `That crop is not in this review queue: ${loc.reason}`
            : 'That crop is not in this review queue (it may already be reviewed).',
        );
        return;
      }
      while (queue.loadedPages < loc.page && queue.hasMore) {
        await queue.loadMore();
      }
      const idx = queue.items.findIndex((i) => i.id === cropId);
      cursor =
        idx >= 0 ? idx : Math.max(0, Math.min(loc.rank ?? 0, queue.items.length - 1));
    } catch (e) {
      toastStore.error(`Locate failed: ${(e as Error).message}`);
    } finally {
      jumpingToCrop = false;
      pendingCropId = null;
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
  // hotkey works everywhere it makes sense. The plates tab is a
  // different flow (confirming a bbox, not a class) so we no-op there
  // and leave the letters free for plate actions.
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
  // for the same tab/preset/classFilter/subjectScope/minBlurRatio values
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
    void classFilter;
    void subjectScope;
    void minBlurRatio;
    const key = JSON.stringify([tab, preset, classFilter, subjectScope, minBlurRatio]);
    if (key === lastImmediateKey) return;
    lastImmediateKey = key;
    // effectiveTab/class_id changes invalidate any in-progress diverse
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
    void plateTextQuery;
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
    const key = JSON.stringify([
      sourceFilter,
      plateTextQuery,
      confMin,
      confMax,
      strategyBar.sort,
      strategyBar.minMistakenness,
      strategyBar.hideNearDuplicates,
      strategyBar.k,
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

  const topClasses = $derived(classesStore.topNForCluster(0, 10));

  // Non-deprecated classes for the filter dropdown (P2-1). classesStore.classes
  // is unfiltered; every other class-offering surface in the app already
  // excludes deprecated (ClassSidebar.svelte:111, ClassSubsetPicker.svelte:35,
  // ShortcutOverlay.svelte:15) — this dropdown was the one that didn't.
  const filterableClasses = $derived(classesStore.classes.filter((c) => !c.deprecated));

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

  const pickerResults = $derived(searchClasses(classesStore.classes, pickerQuery, 50));

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
   * Optimistically drop an item from the queue and advance.
   *
   * Returns the undo closure that puts it back at the same index with the
   * same cursor. EVERY caller must invoke it when the API call fails —
   * otherwise the item vanishes from the operator's queue while the server
   * still holds it unchanged, and it is never seen again this session.
   */
  function _removeFromQueue(item: ReviewItem): () => void {
    const found = queue.items.findIndex((x) => x.id === item.id);
    const removedIdx = found >= 0 ? found : cursor;
    const priorCursor = cursor;
    queue.items = queue.items.filter((x) => x.id !== item.id);
    queue.total = Math.max(0, queue.total - 1);
    cursor = Math.min(cursor, Math.max(0, queue.items.length - 1));
    handledIds.add(item.id);
    maybePrefetch();
    return () => {
      handledIds.delete(item.id);
      const at = Math.min(removedIdx, queue.items.length);
      queue.items = [...queue.items.slice(0, at), item, ...queue.items.slice(at)];
      queue.total += 1;
      cursor = priorCursor;
    };
  }

  async function assign(classId: number): Promise<void> {
    if (!current) return;
    const item = current;
    const cls = classesStore.byId(classId);
    // Optimistic: drop from list and advance.
    const restore = _removeFromQueue(item);
    try {
      await putCropLabel(item.id, classId);
      undoStore.recordWrites([item.id]);
      toastStore.success(`Labeled "${cls?.name ?? classId}".`);
    } catch (e) {
      restore();
      toastStore.error(`Label failed: ${(e as Error).message}`);
    }
  }

  async function confirmAndAdvance(): Promise<void> {
    if (!current) return;
    const proposed = resolveConfirmClassId(current);
    if (proposed == null) {
      // Enter's registration below already routes here vs. openPicker()
      // based on canConfirm, so this only fires from the Confirm button —
      // which is disabled in this state — or a stale click race. Keep the
      // toast as a safety net either way.
      toastStore.warn('No proposed class on this item — press / to search.');
      return;
    }
    await assign(proposed);
  }

  /** "Accept model's class" (item 14, 2026-09-24 logic-moves): the
   *  model_disagreements tab's probe prediction now carries its own
   *  `probe_pred_class_id`, so this assigns it directly — no name→id
   *  lookup needed (closes G4). */
  async function acceptModelClass(): Promise<void> {
    if (!current || current.probe_pred_class_id == null) return;
    await assign(current.probe_pred_class_id);
  }

  function skip(): void {
    cursor = Math.min(queue.items.length - 1, cursor + 1);
    maybePrefetch();
  }

  async function discard(): Promise<void> {
    if (!current) return;
    // Discard = "permanently dismiss this crop from every review queue."
    // Stamps review_dismissed_at on the backend; the review queue's
    // must_not filter excludes any crop with that field set. The crop's
    // class / plate state is left intact — this is NOT an unlabel.
    //
    // Deliberately does NOT push an undoStore entry: Z undoes a *label*
    // write (PUT the prior class, or restore the model suggestion), which
    // is a different action from a dismiss. Reversing a dismiss instead
    // goes through reviewUndismissCrop, surfaced via the "Dismissed"
    // panel below — see openDismissedPanel/undismiss.
    const item = current;
    const restore = _removeFromQueue(item);
    try {
      await reviewDismissCrop(item.id);
      toastStore.success('Dismissed from review (permanent).');
    } catch (e) {
      restore();
      toastStore.error(`Discard failed: ${(e as Error).message}`);
    }
  }

  // -- dismissed-crops panel (un-dismiss) --------------------------------
  // Minimal reachable-from-/review surface for reviewUndismissCrop: a
  // toggleable panel listing crops discard() has dismissed, each with a
  // Restore action. Loaded lazily (only when opened), not kept in sync
  // with the live queue.
  let dismissedPanelOpen = $state<boolean>(false);
  let dismissedItems = $state<Crop[]>([]);
  let dismissedLoading = $state<boolean>(false);

  async function toggleDismissedPanel(): Promise<void> {
    dismissedPanelOpen = !dismissedPanelOpen;
    if (!dismissedPanelOpen) return;
    dismissedLoading = true;
    try {
      const res = await getCrops({ review_dismissed: true, limit: 60, sort: 'recent' });
      dismissedItems = res.items;
    } catch (e) {
      toastStore.error(`Load dismissed crops failed: ${(e as Error).message}`);
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
      toastStore.error(`Restore failed: ${(e as Error).message}`);
    }
  }

  // -- slot-tab actions -------------------------------------------------
  // Inline editor — no modal. The canvas is always live; if the user
  // tweaks the proposed bbox, Confirm saves the edited version. If they
  // leave it alone, Confirm saves the proposal as-is. The goal is one
  // keystroke (Enter) per item when scanning thousands of crops.
  //
  // editedSlotBox lives in the *crop-local* (parent) frame (the same
  // space BboxCanvas operates in) — read straight off readSlot's own
  // projection (slotOf(current, activeSlot)?.subBox?.parent), never
  // re-derived by hand. The seeding effect re-runs whenever the cursor
  // advances to a new crop.
  let editedSlotBox = $state<BBoxNorm | null>(null);
  let slotCanvas = $state<{ handleKey: (e: KeyboardEvent) => boolean } | null>(null);
  // Read-only by default: the canvas only becomes interactive when the
  // operator presses E (or clicks Edit bbox). Most cascade-detected
  // proposals are already correct — forcing the heavy drag-handle UI on
  // every crop is what made the tab feel "weird" vs. the other review
  // tabs. Edit mode resets to false on every cursor advance so the
  // operator always lands on the next item in scan-and-confirm mode.
  let editMode = $state<boolean>(false);
  let slotSaving = $state<boolean>(false);

  // Inline editors for the slot metadata fields. Seeded from the
  // current crop's SlotData on every cursor advance; saved on blur /
  // Enter via patchSlotMeta (PATCH {slot's own patchMeta endpoint}).
  // Each field saves independently with optimistic-UI + revert-on-error,
  // matching the assign() pattern.
  let editedSlotText = $state<string>('');
  let editedSlotStatus = $state<string>('');
  let editedRejectionReason = $state<string>('');
  // Status values an operator is allowed to write, for the ACTIVE slot —
  // closes Finding D (the panel used to render licensePlateSlot's own
  // vocabulary regardless of which slot tab was active). Order matches
  // the deployment's served `GET {API_PREFIX}/regions/statuses` vocabulary
  // when loaded, falling back to the active slot's own
  // `capabilities.lifecycle.states`.
  const slotStatusOptions = $derived(
    activeSlot ? humanWritableStates(activeSlot, regionStatusesStore.list) : [],
  );
  const slotLabels = $derived(activeSlot ? panelLabels(activeSlot) : null);

  // Undo stack for slot confirm/reject. Each entry holds the previously
  // confirmed box so "Back" can re-insert the crop into the queue and
  // restore what the user just saved (allowing them to fix a mistake
  // without re-finding the crop). Bounded to 20 entries — enough for
  // half a session of confusion, small enough to keep memory tiny.
  interface SlotUndoEntry {
    item: ReviewItem;
    insertAt: number;
    /**
     * The sub-box, in the parent-crop-normalized frame, that was sent to
     * the server for this confirm (frame: 'parent') — null means
     * "rejected" (not visible).
     */
    saved: [number, number, number, number] | null;
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
      toastStore.info('Nothing to go back to.');
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
      toastStore.warn(
        `Re-fetch failed; restoring local snapshot: ${(e as Error).message}`,
      );
      fresh = last.item;
    }
    const insertAt = Math.min(last.insertAt, queue.items.length);
    queue.items = reinsertAt(queue.items, last.insertAt, fresh);
    queue.total = queue.total + 1;
    cursor = insertAt;
    toastStore.info('Stepped back. Press E to re-edit, Enter to re-confirm.');
  }

  function _seedSlotFromCurrent(): void {
    editedSlotBox =
      current && activeSlot
        ? (slotOf(current, activeSlot)?.subBox?.parent ?? null)
        : null;
  }

  // Slot-centered viewport for the right-side canvas. **Frozen** —
  // computed once when the crop loads and held steady during edits.
  // If we derived it from `editedSlotBox` instead, every drag tick
  // would recompute the zoom and the IMG transform would pan/scale
  // along with the resize handle, making the box feel like it's
  // rubber-banding the whole image. The canvas applies viewBox as a
  // pure display transform; saved coordinates remain in crop-local
  // frame and project to the slot's stored frame on confirm.
  const SLOT_VIEW_PADDING = 2.5;
  let slotViewBox = $state<BBoxNorm | null>(null);
  function _seedViewBox(): void {
    // Padding/squaring/clamping math lives in viewBox.ts (Phase 0 seam),
    // with its own unit tests; the untrack()-wrapped call site (below)
    // is what actually makes this "frozen" and has to stay here.
    slotViewBox = computeViewBox(editedSlotBox, SLOT_VIEW_PADDING);
  }

  // Reseed whenever the cursor changes (advancing to next crop) or the
  // tab/items reset. Also exit edit mode so the next item lands in
  // read-only scan mode regardless of where we left the previous one.
  $effect(() => {
    void current?.id;
    _seedSlotFromCurrent();
    // Freeze the zoom viewport on the just-seeded bbox. Wrapped in
    // untrack() so the read of `editedSlotBox` inside _seedViewBox
    // does NOT make this effect re-run on every drag tick — that
    // would re-fire _seedSlotFromCurrent and overwrite the user's
    // in-progress resize with the server snapshot ("can't edit the
    // bbox" bug).
    untrack(() => _seedViewBox());
    const seedData = current && activeSlot ? slotOf(current, activeSlot) : null;
    editedSlotText = seedData?.text?.value ?? '';
    editedSlotStatus = seedData?.lifecycle?.status ?? '';
    editedRejectionReason = seedData?.lifecycle?.rejectionReason ?? '';
    editMode = false;
  });

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
        editedSlotText = seedData?.text?.value ?? '';
        editedSlotStatus = seedData?.lifecycle?.status ?? '';
        editedRejectionReason = seedData?.lifecycle?.rejectionReason ?? '';
      }
      toastStore.error(`Save failed: ${(e as Error).message}`);
    } finally {
      slotMetaAborts.finish(id, ac);
    }
  }

  async function commitSlotText(): Promise<void> {
    if (!current || !activeSlot) return;
    const slotData = slotOf(current, activeSlot);
    const next = editedSlotText.trim() || null;
    if ((slotData?.text?.value ?? null) === next) return;
    await saveSlotMeta({ text: next });
  }

  async function commitSlotStatus(): Promise<void> {
    if (!current || !activeSlot) return;
    if (!editedSlotStatus) return;
    const slotData = slotOf(current, activeSlot);
    if (editedSlotStatus === slotData?.lifecycle?.status) return;
    const id = current.id;
    // The reject state implies the bbox is gone — call setSlotBox null
    // to keep the bbox + status in sync (avoids the contradiction of a
    // reject status with a populated sub-box).
    if (statusClearsBox(activeSlot, editedSlotStatus, regionStatusesStore.list)) {
      try {
        const item = await setSlotBox(activeSlot, id, null);
        const idx = queue.items.findIndex((x) => x.id === id);
        if (idx >= 0) queue.items[idx] = { ...queue.items[idx], ...item } as ReviewItem;
        editedSlotBox = null;
      } catch (e) {
        toastStore.error(`Save failed: ${(e as Error).message}`);
      }
      return;
    }
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
    if (!current) return;
    if (editMode) {
      // Cancel-style exit: drop local edits and reseed from server state.
      _seedSlotFromCurrent();
      _seedViewBox();
      editMode = false;
      return;
    }
    // Re-center the zoom on whatever bbox we're about to edit (could
    // differ from the cursor-advance snapshot if the user already saved
    // once on this crop and is re-editing).
    _seedViewBox();
    editMode = true;
  }

  /** [x1,y1,x2,y2] of `box` (a parent-crop-normalized BBoxNorm), for the
   *  `frame: 'parent'` write path — no projection through the parent
   *  vehicle bbox needed; the server does that itself. */
  function _parentFrameTuple(box: BBoxNorm): [number, number, number, number] {
    return [
      box.cx - box.w / 2,
      box.cy - box.h / 2,
      box.cx + box.w / 2,
      box.cy + box.h / 2,
    ];
  }

  async function saveBboxAndExit(): Promise<void> {
    if (!current || !activeSlot) return;
    if (!editedSlotBox) {
      toastStore.warn('No bbox to save — draw one or press Backspace to clear.');
      return;
    }
    const id = current.id;
    const tuple = _parentFrameTuple(editedSlotBox);
    slotSaving = true;
    try {
      const item = await setSlotBox(activeSlot, id, tuple, 'parent');
      const idx = queue.items.findIndex((x) => x.id === id);
      if (idx >= 0) queue.items[idx] = { ...queue.items[idx], ...item } as ReviewItem;
      editMode = false;
      toastStore.success('Bbox saved.');
    } catch (e) {
      toastStore.error(`Save failed: ${(e as Error).message}`);
    } finally {
      slotSaving = false;
    }
  }

  async function confirmSlot(): Promise<void> {
    if (!current || !activeSlot) return;
    if (!editedSlotBox) {
      toastStore.warn(
        `No ${activeSlot.label.singular} bbox to confirm — drag one in or press D to reject.`,
      );
      return;
    }
    const item = current;
    const tuple = _parentFrameTuple(editedSlotBox);
    // Snapshot for "Back" before mutating the queue.
    const undoEntry: SlotUndoEntry = { item, insertAt: cursor, saved: tuple };
    _pushSlotUndo(undoEntry);
    const restore = _removeFromQueue(item);
    try {
      await setSlotBox(activeSlot, item.id, tuple, 'parent');
      toastStore.success(`${activeSlot.label.title} confirmed. ← to go back.`);
    } catch (e) {
      _removeSlotUndo(undoEntry);
      restore();
      toastStore.error(`Confirm failed: ${(e as Error).message}`);
    }
  }

  async function rejectSlot(): Promise<void> {
    if (!current || !activeSlot) return;
    const item = current;
    const undoEntry: SlotUndoEntry = { item, insertAt: cursor, saved: null };
    _pushSlotUndo(undoEntry);
    const restore = _removeFromQueue(item);
    try {
      // null bbox = "not visible" per setSlotBox's clear contract.
      await setSlotBox(activeSlot, item.id, null);
      toastStore.success(`${activeSlot.label.title} rejected. ← to go back.`);
    } catch (e) {
      _removeSlotUndo(undoEntry);
      restore();
      toastStore.error(`Reject failed: ${(e as Error).message}`);
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
    const undoEntry: SlotUndoEntry = { item, insertAt: cursor, saved: null };
    _pushSlotUndo(undoEntry);
    const restore = _removeFromQueue(item);
    try {
      await patchSlotMeta(activeSlot, item.id, { status: fpState });
      toastStore.success('Marked false positive (box kept). ← to go back.');
    } catch (e) {
      _removeSlotUndo(undoEntry);
      restore();
      toastStore.error(`Mark FP failed: ${(e as Error).message}`);
    }
  }

  async function undoLast(): Promise<void> {
    const crop = await undoStore.undoLast();
    if (!crop) return;
    // The item was removed from the queue by assign/discard, so re-insert
    // the restored item at the cursor so the operator can see (and
    // re-verify) what the undo brought back.
    handledIds.delete(crop.id);
    // `crop` (from POST {API_PREFIX}/crops/{id}/label/undo) already carries its
    // own served proposed_class_id/_name (item 11, 2026-09-24
    // logic-moves) — no client fill-in. `reason` is the only field this
    // page adds; every other queue-only field has no meaningful value
    // for a restored item.
    const restored: ReviewItem = {
      ...crop,
      reason: 'restored by undo',
    };
    const without = queue.items.filter((it) => it.id !== crop.id);
    const at = Math.min(cursor, without.length);
    queue.total += without.length === queue.items.length ? 1 : 0;
    queue.items = [...without.slice(0, at), restored, ...without.slice(at)];
    cursor = at;
  }

  // Keyboard shortcuts. Per-class letter hotkeys (configured on /classes)
  // are routed through dropOnClassStore by the layout-level keydown
  // listener and work on every tab. The shortcuts below are the
  // tab-action shortcuts; on the plates tab Enter/D get rebound to plate
  // confirm/reject so the same finger pattern works for both flows.
  //
  // On the plates tab, behavior splits between read-only scan mode
  // (default) and edit mode (operator pressed E or Edit bbox):
  //   - read-only: arrows page the queue, Enter confirms-and-advances,
  //     E enters edit mode — matches the other review tabs.
  //   - edit:      arrows nudge the bbox, Enter saves+exits edit mode,
  //     Esc cancels edit, the bbox canvas owns the keystroke flow.
  $effect(() => {
    const offs: Array<() => void> = [];
    const reg = (combo: string, fn: () => void | Promise<void>, desc: string) =>
      offs.push(keyboardStore.register(combo, () => void fn(), 'review', desc));

    if (activeSlot) {
      // Table built by the slot-generic slotKeymap module (P2.8c), reading
      // activeSlot.capabilities.queue.keymap instead of a second
      // hand-maintained copy — asserted by slotKeymap.test.ts rather than
      // only readable here.
      for (const entry of buildSlotKeymap(activeSlot, editMode, {
        confirm: confirmSlot,
        reject: rejectSlot,
        markFalsePositive,
        toggleEdit,
        back: slotBack,
        advance: () => {
          cursor = Math.min(queue.items.length - 1, cursor + 1);
          maybePrefetch();
        },
        saveAndExit: saveBboxAndExit,
      })) {
        reg(entry.combo, entry.fn, entry.description);
      }
    } else {
      reg(
        'enter',
        () => {
          // P1-5: a blank proposal made Enter a silent no-op. Open the
          // class picker instead so the operator can act in one keystroke
          // rather than hitting Enter and wondering why nothing happened.
          if (canConfirm) return confirmAndAdvance();
          openPicker();
        },
        'Confirm proposed & advance (or search classes if blank)',
      );
      reg('d', discard, 'Discard');
      reg('/', openPicker, 'Search all classes…');
    }
    reg('n', skip, 'Skip');
    reg('z', undoLast, 'Undo last');

    let canvasKey: ((e: KeyboardEvent) => void) | null = null;
    if (activeSlot?.capabilities.subBox != null && editMode) {
      // Edit mode only: forward bbox-fine-tune keys (arrows, [ / ],
      // Backspace) into the slot's bbox canvas. Outside edit mode arrows
      // page the queue like every other tab.
      canvasKey = (e: KeyboardEvent) => {
        if (!slotCanvas) return;
        const target = e.target as HTMLElement | null;
        if (target && /^(input|textarea|select)$/i.test(target.tagName)) return;
        if (slotCanvas.handleKey(e)) e.preventDefault();
      };
      window.addEventListener('keydown', canvasKey);
    } else if (!isSlotTab(tab)) {
      // On non-slot tabs arrow keys navigate the queue.
      reg(
        'arrowleft',
        () => {
          cursor = Math.max(0, cursor - 1);
        },
        'Previous item',
      );
      reg(
        'arrowright',
        () => {
          cursor = Math.min(queue.items.length - 1, cursor + 1);
          maybePrefetch();
        },
        'Next item',
      );
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
    <div class="flex min-w-0 grow items-center gap-1 overflow-x-auto whitespace-nowrap">
      {#each REVIEW_TABS as t (t.id)}
        <button
          type="button"
          class="shrink-0 px-3 py-2.5 text-sm border-b-2 {tab === t.id
            ? 'border-blue-500 text-white'
            : 'border-transparent text-zinc-400 hover:text-zinc-200'}"
          onclick={() => {
            tab = t.id;
            pendingCropId = null;
            const url = new URL(page.url);
            url.searchParams.set('tab', t.urlId);
            url.searchParams.delete('crop_id');
            replaceState(url, {});
            // Presets only make sense on the All tab — switching to any
            // other tab (or re-landing on All from one) always starts
            // from plain All rather than silently carrying a stale chip.
            preset = null;
            closePicker();
          }}
        >
          {t.label}
        </button>
      {/each}
    </div>
    <span
      data-testid="queue-counter"
      class="shrink-0 pl-2 font-mono text-xs text-zinc-500"
    >
      {queue.items.length > 0 ? `${cursor + 1} / ${queue.items.length}` : '—'} loaded · {queue.total}
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
      diverseKDefault={DIVERSE_K_DEFAULT}
      diverseKMax={DIVERSE_K_MAX}
      diverseMeta={diverseSelection
        ? {
            method: diverseSelection.method,
            version: diverseSelection.version,
            n_pool: diverseSelection.n_pool,
          }
        : null}
    />
    {#if sortFallbackReason}
      <span
        class="text-[11px] text-amber-300"
        title="The requested sort couldn't be honored server-side; showing the default order instead."
      >
        sort fallback: {sortFallbackReason}
      </span>
    {/if}
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
    <label class="flex shrink-0 items-center gap-1.5">
      <span class="text-zinc-400">Source</span>
      <input
        type="text"
        bind:value={sourceFilter}
        placeholder="any"
        class="input-sm w-32"
      />
    </label>

    <label class="flex shrink-0 items-center gap-1.5">
      <span class="text-zinc-400">Class</span>
      <select bind:value={classFilter} class="select-sm">
        <option value={null}>any</option>
        {#each filterableClasses as cls (cls.id)}
          <option value={cls.id}>{cls.name}</option>
        {/each}
      </select>
    </label>

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
        class="input-sm w-16"
      />
    </label>

    {#if activeSlot?.capabilities.queue?.textFilter}
      <label
        class="flex shrink-0 items-center gap-1.5"
        class:opacity-40={diverseMode}
        title={diverseMode ? 'not applied to diverse selection' : undefined}
      >
        <span class="text-zinc-400">{activeSlot.capabilities.queue.textFilter.label}</span
        >
        <input
          type="text"
          bind:value={plateTextQuery}
          disabled={diverseMode}
          placeholder={activeSlot.capabilities.queue.textFilter.placeholder}
          class="input-sm w-28"
        />
      </label>
    {/if}

    <!-- Always available, on every tab and preset — matches Conf/Class/HDD
         source below/above, and the backend's own query builder already
         treats max_rank/min_blur_ratio as tab-agnostic. Used to be gated to
         only primary_low_conf/coco_blind_spots, which made these controls
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
      <SubjectScopeToggle
        bind:value={subjectScope}
        labels={['Top 2', 'Largest', '+2nd']}
        label="subject"
      />
      <BlurSlider
        bind:value={blurSlider}
        oncommit={commitBlur}
        max={BLUR_MAX}
        title="Hide crops blurrier than this"
      />
    </div>

    {#if tab === 'all'}
      <!-- Quick-filter preset chips (2026-09 tab consolidation) — Mismatches
           / Gemma Low-Conf / Primary·Low-Conf collapsed from top-level tabs
           into these, since live counts showed each was too big (11-97% of
           the dataset) to be a curated queue. Each chip reuses that former
           tab's exact backend query unchanged (see $lib/reviewTabs.ts);
           radio-style — picking a second chip swaps the first, clicking the
           active one (or "clear") returns to plain All. -->
      <div class="flex shrink-0 flex-wrap items-center gap-1.5">
        <span class="text-zinc-400">Quick filter</span>
        {#each REVIEW_PRESETS as p (p.id)}
          <button
            type="button"
            class="chip {preset === p.id
              ? 'border-blue-500/60 bg-blue-500/15 text-blue-100'
              : 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800'}"
            aria-pressed={preset === p.id}
            title={p.description}
            onclick={() => togglePreset(p.id)}
          >
            {p.label}
          </button>
        {/each}
        {#if preset}
          <button
            type="button"
            class="chip bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
            onclick={() => (preset = null)}
          >
            clear
          </button>
        {/if}
      </div>
    {/if}

    <span class="grow"></span>

    <span class="hidden text-[11px] text-zinc-500 md:inline">
      {#if activeSlot?.capabilities.subBox && editMode}
        <kbd>↑↓←→</kbd> nudge · <kbd>[ ]</kbd> right edge · <kbd>Enter</kbd> save ·
        <kbd>Esc</kbd> cancel
      {:else if activeSlot}
        <kbd>Enter</kbd> confirm · <kbd>{rejectKeyGlyph(activeSlot)}</kbd> reject
        {#if activeSlot.capabilities.lifecycle?.falsePositiveState}
          · <kbd>F</kbd> false-pos
        {/if}
        {#if activeSlot.capabilities.subBox}
          · <kbd>E</kbd> edit
        {/if}
        · <kbd>N</kbd> skip · <kbd>←</kbd> back
      {:else}
        per-class letter assigns · <kbd>/</kbd> search all classes ·
        <kbd>Enter</kbd> confirm · <kbd>N</kbd> skip · <kbd>D</kbd> discard ·
        <kbd>Z</kbd> undo
      {/if}
    </span>
  </div>

  <!-- Body -->
  <div class="grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-hidden p-4 lg:grid-cols-2">
    {#if queue.loading && queue.items.length === 0}
      <p class="col-span-full text-sm text-zinc-500">Loading...</p>
    {:else if queue.error}
      <p class="col-span-full text-sm text-red-300">API unavailable: {queue.error}</p>
    {:else if !current}
      <p class="col-span-full text-sm text-zinc-500">Queue empty.</p>
    {:else}
      <!-- Source image with bbox -->
      <div class="flex min-h-0 flex-col surface p-2">
        <div class="mb-2 flex items-center gap-2 px-1 text-xs text-zinc-400">
          <span>source</span>
          <span class="grow"></span>
          <span class="font-mono">{current.source ?? ''}</span>
        </div>
        <div class="flex min-h-0 flex-1 items-center justify-center bg-zinc-950">
          <img
            src={getSourceImageWithBbox(
              current.id,
              1280,
              // Cache-bust on sub-box edits so the burned-in overlay
              // refreshes after a save. updated_at would be nicer but
              // not every code path mutates it locally; the raw xyxy
              // tuple is a stable enough fingerprint.
              (activeSlot ? slotOf(current, activeSlot)?.subBox?.rawXyxy : null)?.join(
                ',',
              ) ?? 'none',
            )}
            alt="source"
            loading="lazy"
            decoding="async"
            class="max-h-full max-w-full object-contain"
          />
        </div>
      </div>

      <!-- Crop + meta -->
      <div class="flex min-h-0 flex-col surface p-2">
        <div class="mb-2 flex items-center gap-2 px-1 text-xs text-zinc-400">
          <span>crop</span>
          <span class="grow"></span>
          <span class="font-mono">{current.id.slice(0, 12)}…</span>
        </div>
        <div class="flex min-h-0 flex-1 items-center justify-center bg-zinc-950">
          {#if activeSlot?.capabilities.subBox && editMode}
            <!-- Edit mode — drag/resize the proposal directly, then hit
                 Enter to save. Square aspect keeps the canvas math
                 stable; the read-only default below shows the crop at
                 natural aspect to match the other review tabs. -->
            <BboxCanvas
              bind:this={slotCanvas}
              cropId={current.id}
              bind:bbox={editedSlotBox}
              viewBox={slotViewBox}
              busy={slotSaving}
              label={activeSlot.label.title}
              class="aspect-square w-auto h-full max-h-full min-w-0 max-w-full"
            />
          {:else if activeSlot?.capabilities.subBox}
            <!-- Read-only default: same <img> layout as every other tab,
                 with a thin yellow ring overlay on the proposed bbox.
                 No grabbable handles, no pointer capture — the bbox is
                 just shown. Press E to edit. -->
            <BboxCanvas
              cropId={current.id}
              bbox={editedSlotBox}
              viewBox={slotViewBox}
              readonly
              label={activeSlot.label.title}
              class="aspect-square w-auto h-full max-h-full min-w-0 max-w-full"
            />
          {:else}
            <img
              src={getThumbUrl(current.id, 384)}
              alt="crop"
              loading="lazy"
              decoding="async"
              class="max-h-full max-w-full object-contain"
            />
          {/if}
        </div>

        <dl class="mt-3 grid grid-cols-2 gap-y-1 text-xs">
          <dt class="text-zinc-500">Reason</dt>
          <dd class="text-zinc-200">{current.reason}</dd>

          <dt class="text-zinc-500">Current label</dt>
          <dd class="text-zinc-200">
            {current.class_name ?? '—'}
            <span class="ml-1 text-zinc-500">({current.label_source})</span>
          </dd>

          <dt class="text-zinc-500">Proposed</dt>
          <dd class="text-yellow-200">{current.proposed_class_name ?? '—'}</dd>

          {#if current.probe_pred_class}
            <!-- G4 closed 2026-09-24 (logic-moves item 14): the backend
                 now serves `probe_pred_class_id` alongside the display
                 name, so "Accept" no longer needs a client-side
                 name→id lookup — assign() takes the served id directly. -->
            <dt class="text-zinc-500">Model predicts</dt>
            <dd class="flex flex-wrap items-center gap-1.5 text-zinc-200">
              {current.probe_pred_class}
              {#if current.probe_pred_entropy != null}
                <ScoreChip label="entropy" value={current.probe_pred_entropy} size="sm" />
              {/if}
              {#if current.probe_pred_class_id != null && current.probe_pred_class_id !== current.class_id}
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

          <dt class="text-zinc-500">Confidence</dt>
          <dd class="font-mono">
            {current.label_confidence != null
              ? `${(current.label_confidence * 100).toFixed(1)}%`
              : '—'}
          </dd>

          {#if current.proposal_name}
            <dt class="text-zinc-500">Proposal hint</dt>
            <dd>
              <span
                class="rounded border border-cyan-500/40 bg-cyan-500/15 px-1.5 py-0.5 text-[11px] text-cyan-200"
                title="COCO YOLO11 detected a vehicle here that v6 missed. Coarse class — pick the make below (bicycle/motorcycle/boat may be near one-click)."
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
              {slotData?.subBox?.score != null
                ? `${(slotData.subBox.score * 100).toFixed(1)}%`
                : '—'}
            </span>
            <span class="text-zinc-500">{slotLabels.statusLabel}</span>
            <span>
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
            </span>
            <span class="text-zinc-500">Detector</span>
            <span class="flex flex-wrap items-center gap-1.5">
              {#if slotData?.provenance?.detector}
                <ProvenanceChip
                  detector={slotData.provenance.detector}
                  version={slotData.provenance.detectorVersion}
                />
                {#if slotData.provenance.verifier}
                  <ProvenanceChip
                    detector={slotData.provenance.verifier}
                    tag="verify"
                    version={slotData.provenance.verifierVersion}
                    size="sm"
                  />
                {/if}
              {:else}
                <span class="text-zinc-500">—</span>
              {/if}
              {#if current.mistakenness_score != null}
                <ScoreChip
                  label="mistakenness"
                  value={current.mistakenness_score}
                  method={current.mistakenness_method}
                  version={current.mistakenness_version}
                  size="sm"
                />
              {/if}
              {#if !editedSlotBox && !editMode}
                <span
                  class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-[10px] text-zinc-400"
                  title={slotLabels.noBoxHint}
                >
                  no bbox · press E to draw
                </span>
              {/if}
            </span>
            {#if slotData?.provenance?.chain && slotData.provenance.chain.length > 0}
              <span class="text-zinc-500">Cascade</span>
              <span class="flex flex-wrap items-center gap-1">
                {#each slotData.provenance.chain as entry (entry)}
                  <ProvenanceChip raw={entry} size="sm" />
                {/each}
              </span>
            {/if}
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
                spellcheck="false"
                autocapitalize={activeSlot.capabilities.text?.transform === 'uppercase'
                  ? 'characters'
                  : 'off'}
                class="w-28 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-xs text-zinc-100 focus:border-blue-500 focus:outline-none {activeSlot
                  .capabilities.text?.monospace
                  ? 'font-mono'
                  : ''}"
              />
              {#if slotData?.text?.source}
                <ProvenanceChip detector={slotData.text.source} size="sm" />
              {/if}
              {#if slotData?.text?.confidence != null}
                <span class="text-[10px] text-zinc-500">
                  {(slotData.text.confidence * 100).toFixed(0)}%
                </span>
              {/if}
            </span>
            {#if statusWantsRejectionReason(activeSlot, editedSlotStatus, regionStatusesStore.list)}
              <span class="text-zinc-500">Rejection reason</span>
              <span>
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
              </span>
            {/if}
          </div>
          <div class="mt-3 flex flex-wrap gap-2">
            {#if editMode}
              <button
                class="btn btn-primary"
                type="button"
                onclick={saveBboxAndExit}
                disabled={slotSaving}
              >
                Save bbox
              </button>
              <button
                class="btn"
                type="button"
                onclick={toggleEdit}
                disabled={slotSaving}
              >
                Cancel
              </button>
            {:else}
              <button class="btn btn-primary" type="button" onclick={confirmSlot}>
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
                    .singular} — keep the box as a training hard negative (F)"
                >
                  False positive
                </button>
              {/if}
              <button class="btn" type="button" onclick={skip}>Skip</button>
              <button
                class="btn"
                type="button"
                onclick={toggleEdit}
                aria-pressed={editMode}
                title="Toggle bbox edit mode (E)"
              >
                Edit bbox
              </button>
              <button
                class="btn"
                type="button"
                onclick={slotBack}
                disabled={slotUndoStack.length === 0}
                title="Re-open the most-recently confirmed {activeSlot.label
                  .singular} (←)"
              >
                ← Back
              </button>
            {/if}
          </div>
          {#if slotUndoStack.length > 0}
            <p class="mt-1 text-[10px] text-zinc-500">
              {slotUndoStack.length} confirmed in this session — press ← to step back.
            </p>
          {/if}
        {:else}
          <div class="mt-3 flex flex-wrap gap-2">
            <button
              class="btn btn-primary"
              type="button"
              onclick={confirmAndAdvance}
              disabled={!canConfirm}
              title={canConfirm
                ? undefined
                : 'No proposed class on this item — press / or Enter to search.'}
            >
              Confirm
            </button>
            <button class="btn" type="button" onclick={skip}>Skip</button>
            <button class="btn btn-danger" type="button" onclick={discard}>Discard</button
            >
            <button class="btn" type="button" onclick={undoLast}>Undo</button>
          </div>
        {/if}

        <!-- Most-validated classes — click to label OR press the per-class
             hotkey configured on /classes. Hotkey badges only show for
             classes the user has explicitly bound (otherwise the strip is
             still clickable, just no kbd hint). The class strip is hidden
             on the plates tab; class assignment isn't relevant there. -->
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
              title="Search all classes (/)"
              onclick={openPicker}
            >
              <kbd
                class="mr-1.5 rounded bg-zinc-800 px-1 py-0.5 font-mono text-[10px] text-blue-300"
              >
                /
              </kbd>
              search all classes…
            </button>
          </div>
          <p class="mt-1.5 text-[10px] text-zinc-500">
            Click a class, press its bound letter, or press / to search all classes (set
            hotkeys on /classes).
          </p>
        {/if}
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
      {Math.min(cursor + 1, queue.items.length)} / {queue.total}
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

{#if pickerOpen}
  <!-- Class picker (P1-4) — fuzzy-search over every non-deprecated class,
       opened with / or the "search all classes…" button. Backdrop click
       and Esc both close without assigning. -->
  <div
    class="fixed inset-0 z-50 flex items-start justify-center bg-black/50 pt-24"
    onclick={closePicker}
    role="presentation"
  >
    <div
      class="w-full max-w-md overflow-hidden rounded-lg border border-zinc-700 bg-zinc-900 shadow-xl"
      onclick={(e) => e.stopPropagation()}
      role="presentation"
    >
      <input
        bind:this={pickerInputEl}
        bind:value={pickerQuery}
        type="text"
        placeholder="Search all {classesStore.classes.length} classes…"
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
