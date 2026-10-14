/**
 * /train's active-run status poll and the paginated past-runs table.
 * Call during component init (it registers the polling effect); the
 * effect's teardown and `dispose()` stop the interval and abort the
 * in-flight status read on unmount.
 */
import { apiErrorText } from '$lib/api_errors';
import { getTrainRuns, getTrainStatus } from '$lib/api_train';
import type { TrainJobStatus } from '$lib/types_train';

const ACTIVE_STATES: ReadonlySet<string> = new Set([
  'queued',
  'starting',
  'running',
  'exporting',
]);

const RUNS_PAGE = 25;

export function createTrainRunsController() {
  // ---- Active job + status polling -------------------------------------
  let activeStatus = $state<TrainJobStatus | null>(null);
  let statusPoll: ReturnType<typeof setInterval> | null = null;
  let statusAbort: AbortController | null = null;

  // We keep a separate reference to the *most recent* job_id we kicked
  // off ourselves so the log tail and progress panel can stay glued to
  // it across status churn.
  let trackedJobId = $state<string | null>(null);

  const isActive = $derived(activeStatus ? ACTIVE_STATES.has(activeStatus.state) : false);

  async function pullStatus(): Promise<void> {
    statusAbort?.abort();
    statusAbort = new AbortController();
    try {
      const next = trackedJobId
        ? await getTrainStatus(trackedJobId, statusAbort.signal)
        : await getTrainStatus(undefined, statusAbort.signal);
      activeStatus = next;
      if (
        next?.state === 'finished' ||
        next?.state === 'failed' ||
        next?.state === 'cancelled'
      ) {
        // Refresh the past-runs table as soon as the active one
        // terminates so the row appears immediately.
        void refreshRuns();
      }
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
      // Don't toast on every poll failure — only the first.
    }
  }

  // Always poll. Fast (5s) while a run is active so the epoch counter and
  // GPU strip stay live; slow (10s) when idle so a job submitted from
  // somewhere else (curl, another browser tab) shows up automatically.
  // pullStatus() also kicks refreshRuns() on terminal-state transitions
  // so the past-runs table picks up newly-finished jobs without a manual
  // page reload.
  $effect(() => {
    void pullStatus();
    void refreshRuns();
    const interval = isActive ? 5000 : 10000;
    if (statusPoll) clearInterval(statusPoll);
    statusPoll = setInterval(() => {
      void pullStatus();
      if (!isActive) {
        // Idle path: also refresh the past-runs table on every tick so
        // jobs created out-of-band appear without a manual reload.
        void refreshRuns();
      }
    }, interval);
    return () => {
      if (statusPoll) clearInterval(statusPoll);
      statusPoll = null;
      statusAbort?.abort();
    };
  });

  // ---- Past runs (paginated) -------------------------------------------
  let runs = $state<TrainJobStatus[]>([]);
  let runsTotal = $state<number>(0);
  let runsLoading = $state<boolean>(false);
  let runsLoadingMore = $state<boolean>(false);
  let runsError = $state<string | null>(null);
  const runsHasMore = $derived(runs.length < runsTotal);

  async function refreshRuns(): Promise<void> {
    runsLoading = true;
    runsError = null;
    try {
      const res = await getTrainRuns(RUNS_PAGE, 0);
      runs = res.items ?? [];
      runsTotal = res.total ?? runs.length;
    } catch (e) {
      runsError = apiErrorText(e);
    } finally {
      runsLoading = false;
    }
  }

  async function loadMoreRuns(): Promise<void> {
    if (runsLoadingMore || !runsHasMore) return;
    runsLoadingMore = true;
    try {
      const res = await getTrainRuns(RUNS_PAGE, runs.length);
      // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local dedup set consumed synchronously within this call, never stored in reactive state
      const seen = new Set(runs.map((r) => r.job_id));
      const fresh = (res.items ?? []).filter((r) => !seen.has(r.job_id));
      runs = [...runs, ...fresh];
      runsTotal = res.total ?? runsTotal;
    } catch (e) {
      runsError = apiErrorText(e);
    } finally {
      runsLoadingMore = false;
    }
  }

  // ---- Campaigns -------------------------------------------------------
  // Active campaign = the campaign_id on the active run, if any. We
  // group sibling runs from the past-runs feed.
  const activeCampaignId = $derived(activeStatus?.campaign_id ?? null);
  const campaignRuns = $derived.by(() => {
    if (!activeCampaignId) return [] as TrainJobStatus[];
    const live = activeStatus;
    const all = [...runs];
    if (live && !all.some((r) => r.job_id === live.job_id)) {
      all.unshift(live);
    }
    return all.filter((r) => r.campaign_id === activeCampaignId);
  });

  function dispose(): void {
    if (statusPoll) clearInterval(statusPoll);
    statusAbort?.abort();
  }

  return {
    get activeStatus() {
      return activeStatus;
    },
    set trackedJobId(v: string | null) {
      trackedJobId = v;
    },
    get isActive() {
      return isActive;
    },
    get runs() {
      return runs;
    },
    get runsTotal() {
      return runsTotal;
    },
    get runsLoading() {
      return runsLoading;
    },
    get runsLoadingMore() {
      return runsLoadingMore;
    },
    get runsError() {
      return runsError;
    },
    get runsHasMore() {
      return runsHasMore;
    },
    get activeCampaignId() {
      return activeCampaignId;
    },
    get campaignRuns() {
      return campaignRuns;
    },
    pullStatus,
    refreshRuns,
    loadMoreRuns,
    dispose,
  };
}

export type TrainRunsController = ReturnType<typeof createTrainRunsController>;
