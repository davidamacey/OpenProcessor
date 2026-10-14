/**
 * `/review` strategy bar and its diverse overlay (P2-10, pool-scale
 * k-center-greedy selection): the sort/filter strategy, the pinned sort
 * default, and the diverse-selection job (start, poll, cancel, page through
 * the selected ids).
 *
 * Moved verbatim out of `review/+page.svelte`. The selection scope reads the
 * page's Source box and class filter through `scope`, a getter, because the
 * filter controller is built after this one (it reads `diverseMode`).
 */

import { onMount } from 'svelte';
import {
  apiErrorText,
  cancelSelect,
  getCrop,
  getSelectStatus,
  selectDiverse,
} from '$lib/api';
import { endpointForTab, tabHonorsPinnedSortDefault } from '$lib/reviewTabs';
import { isDiverseOverlayAvailable } from '$lib/strategies';
import { createStrategyBar } from '$lib/strategyBar.svelte';
import { REVIEW_PAGE_SIZE as pageSize } from '$lib/review/reviewQueueLoader.svelte';
import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
import type { DiverseSelection, ReviewItem } from '$lib/types';
import { curationSettingsStore } from '$stores/curationSettings.svelte';
import { strategiesStore } from '$stores/strategies.svelte';

export interface ReviewDiverseOptions {
  tabs: ReviewTabsController;
  /** The term filters POST /select/diverse accepts, read when a selection starts. */
  scope: () => { sourceFilter: string; classNames: readonly string[] };
}

export function createReviewDiverseController(opts: ReviewDiverseOptions) {
  const { tabs } = opts;
  const effectiveTab = $derived(tabs.effectiveTab);

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
    const { sourceFilter, classNames } = opts.scope();
    // Server-side, scope.filters on POST {API_PREFIX}/select/diverse only supports
    // term/terms filters (class_name, source) — NOT conf_min/conf_max/
    // min_blur_ratio/max_rank/region text. Those controls are disabled in
    // the UI while diverseMode is active (see the filter bar below) so
    // this never silently drops something the operator thinks is applied.
    const f: Record<string, unknown> = {};
    if (sourceFilter) f.source = sourceFilter;
    // scope.filters are exact-match terms keyed by index field; a list
    // becomes a `terms` clause.
    if (classNames.length > 0) f.class_name = [...classNames];
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

  onMount(() => stopDiversePolling);

  return {
    strategyBar,
    get pinnedSortId() {
      return pinnedSortId;
    },
    get diverseAvailable() {
      return diverseAvailable;
    },
    get diverseMode() {
      return diverseMode;
    },
    DIVERSE_K_DEFAULT,
    get diverseKMax() {
      return diverseKMax;
    },
    get diverseSelection() {
      return diverseSelection;
    },
    set diverseSelection(v: DiverseSelection | null) {
      diverseSelection = v;
    },
    get diverseJobId() {
      return diverseJobId;
    },
    get diverseJobStatus() {
      return diverseJobStatus;
    },
    get diverseError() {
      return diverseError;
    },
    cancelDiverseJob,
    fetchDiversePage,
  };
}

export type ReviewDiverseController = ReturnType<typeof createReviewDiverseController>;
