/**
 * "Act on every item a selection names": exclude, restore, label as a
 * class, or move to a cluster, over an `ItemSelection` (a filter or crop ids,
 * plus an optional limit / sample / seed).
 *
 * The count the operator confirms is the server's: opening an action, or
 * changing the limit / sample / seed, runs the same write with
 * `dry_run: true` and shows its `selected`. Confirm repeats it with
 * `dry_run: false` against the filter as it was when the dialog opened.
 * The recorded operation (or, for ignore / restore, the ids the write served)
 * goes to `undoStore` so Z reverts it. Label and move are queued jobs: confirm
 * follows the job and reports the server's counts from its result. Refusals
 * (both/neither of crop ids and filter, an empty filter without a limit,
 * too many items) are the server's own message.
 */
import { getClusters } from '$lib/api_clusters';
import { apiErrorText } from '$lib/api_errors';
import { runLabelOp } from '$lib/labelOps/runLabelOp';
import type { Cluster } from '$lib/types';
import type { JobView } from '$lib/types_jobs';
import type { LabelOpAccepted } from '$lib/types_labelOps';
import {
  batchExcludeSelection,
  batchUnexcludeSelection,
  bulkLabelSelection,
  moveSelectionToCluster,
} from '$lib/api_select';
import { undoStore } from '$stores/undo.svelte';
import { toastStore } from '$stores/toast.svelte';
import type {
  ItemSelection,
  SelectionDryRun,
  SelectionSample,
} from '$lib/types_itemFilter';

export type SelectionActionId = 'exclude' | 'unexclude' | 'label' | 'move';

export interface SelectionActionResult {
  summary: string;
}

type ExcludeBody = {
  excluded?: number;
  unexcluded?: number;
  updated_ids: string[];
};

function isDryRun(r: unknown): r is SelectionDryRun {
  return !!r && typeof r === 'object' && (r as { dry_run?: unknown }).dry_run === true;
}

export interface SelectionActionOptions {
  /** True when the host moves on as soon as the write is queued (the label-then-next flow):
   *  the dialog then closes at once and the job is followed in the background. */
  detachOnQueued?: () => boolean;
  /** The dialog closed while its job runs on (a click on Run in background, or `detachOnQueued`). */
  ondetached?: () => void;
}

export function createSelectionActionController(
  base: () => ItemSelection,
  options: SelectionActionOptions = {},
) {
  let action = $state<SelectionActionId | null>(null);
  let snapshot: ItemSelection = {};
  let classId = $state<number | null>(null);
  /** The served ignore reason an `exclude` action records. */
  let reason = $state('ignore');
  let clusterUid = $state<string | null>(null);
  let clusterOptions = $state<Cluster[]>([]);
  let limit = $state<number | null>(null);
  let sample = $state<SelectionSample | null>(null);
  let seed = $state<number | null>(null);
  let selected = $state<number | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let confirmed = $state(false);
  let result = $state<SelectionActionResult | null>(null);
  /** The served job of a queued label or move: its state, queue position and progress. */
  let job = $state<JobView | null>(null);
  /** The dialog is closed while the job is still followed ("Run in background"). */
  let detached = $state(false);
  let inflight: AbortController | null = null;
  /** Bumped by every `open` and `cancel`: a job followed in the background must not touch
   *  the state of a dialog opened after it. */
  let session = 0;

  function selection(): ItemSelection {
    const sel: ItemSelection = { ...snapshot };
    if (limit != null) sel.limit = limit;
    if (sample != null) sel.sample = sample;
    if (sample === 'random' && seed != null) sel.seed = seed;
    return sel;
  }

  /** The missing field an action needs before it can ask the server. */
  function missingParam(): boolean {
    if (action === 'label') return classId == null;
    if (action === 'move') return clusterUid == null;
    return false;
  }

  function call(dryRun: boolean, signal: AbortSignal): Promise<unknown> {
    const sel = selection();
    switch (action) {
      case 'exclude':
        return batchExcludeSelection(sel, reason, dryRun, signal);
      case 'unexclude':
        return batchUnexcludeSelection(sel, dryRun, signal);
      case 'label':
        return bulkLabelSelection(sel, classId!, dryRun, signal);
      case 'move':
        return moveSelectionToCluster(sel, clusterUid!, dryRun, signal);
      default:
        return Promise.reject(new Error('no action chosen'));
    }
  }

  async function refresh(): Promise<void> {
    if (action == null) return;
    inflight?.abort();
    selected = null;
    error = null;
    if (missingParam()) return;
    const ctrl = new AbortController();
    inflight = ctrl;
    loading = true;
    try {
      const res = await call(true, ctrl.signal);
      if (ctrl.signal.aborted) return;
      selected = isDryRun(res) ? res.selected : null;
    } catch (e) {
      if (ctrl.signal.aborted || (e as Error).name === 'AbortError') return;
      error = apiErrorText(e);
    } finally {
      if (inflight === ctrl) {
        loading = false;
        inflight = null;
      }
    }
  }

  async function open(next: SelectionActionId): Promise<void> {
    session += 1;
    detached = false;
    job = null;
    action = next;
    snapshot = structuredClone($state.snapshot(base()));
    confirmed = false;
    result = null;
    let optionsError: string | null = null;
    if (next === 'move') {
      try {
        // The target must be a cluster of the serving partition: offer the served cards.
        clusterOptions = (await getClusters({ per_cluster: 0 })).items;
      } catch (e) {
        clusterOptions = [];
        optionsError = apiErrorText(e);
      }
    }
    await refresh();
    if (optionsError) error = optionsError;
  }

  const canConfirm = $derived(
    action != null && selected != null && selected > 0 && !loading && !confirmed,
  );

  /** Follows the queued label or move to its end and records its undo; returns the summary. */
  async function followWrite(
    doing: 'label' | 'move',
    accepted: LabelOpAccepted,
    signal: AbortSignal,
    mine: number,
  ): Promise<string> {
    job = null;
    if (options.detachOnQueued?.() && accepted.job_id != null) detach();
    const out = await runLabelOp(accepted, {
      signal,
      onUpdate: (j) => {
        if (mine === session) job = j;
      },
    });
    // A selection names no ids; the page reloads to show the write.
    undoStore.recordLabelOp(out.opId, []);
    const verb = doing === 'label' ? 'Labeled' : 'Moved';
    return out.conflicts > 0
      ? `${verb} ${out.updated} (${out.conflicts} blocked)`
      : `${verb} ${out.updated}`;
  }

  function finishExclusion(doing: SelectionActionId | null, body: ExcludeBody): string {
    if (doing === 'exclude') undoStore.recordExclusion(body.updated_ids);
    else undoStore.recordUnexclusion(body.updated_ids);
    const n = body.excluded ?? body.unexcluded ?? body.updated_ids.length;
    return `${doing === 'exclude' ? 'Ignored' : 'Restored'} ${n}`;
  }

  function toastDone(doing: SelectionActionId | null, summary: string): void {
    const queued = doing === 'label' || doing === 'move';
    toastStore.push({
      kind: 'success',
      text: `${summary}.`,
      ...(queued
        ? {
            ttl_ms: 8000,
            action: { label: 'Undo', run: () => void undoStore.undoLast() },
          }
        : {}),
    });
  }

  async function confirm(): Promise<void> {
    if (!canConfirm) return;
    inflight?.abort();
    const ctrl = new AbortController();
    inflight = ctrl;
    const mine = session;
    const doing = action;
    loading = true;
    error = null;
    try {
      const res = await call(false, ctrl.signal);
      if (isDryRun(res)) {
        error = 'The server answered a dry run; nothing was written.';
        return;
      }
      const summary =
        doing === 'label' || doing === 'move'
          ? await followWrite(doing, res as LabelOpAccepted, ctrl.signal, mine)
          : finishExclusion(doing, res as ExcludeBody);
      toastDone(doing, summary);
      if (mine !== session) return;
      confirmed = true;
      result = { summary };
    } catch (e) {
      if (mine === session) error = apiErrorText(e);
      else toastStore.error(apiErrorText(e));
    } finally {
      if (inflight === ctrl) inflight = null;
      if (mine === session) {
        loading = false;
        if (error) detached = false;
      }
    }
  }

  function detach(): void {
    if (detached) return;
    detached = true;
    options.ondetached?.();
  }

  /** Closes the dialog and lets the queued job run on; its end shows as a toast. */
  function runInBackground(): void {
    if (loading && job) detach();
  }

  function cancel(): void {
    session += 1;
    inflight?.abort();
    inflight = null;
    detached = false;
    job = null;
    action = null;
    selected = null;
    error = null;
    loading = false;
    confirmed = false;
    result = null;
  }

  return {
    get action() {
      return action;
    },
    get selected() {
      return selected;
    },
    get loading() {
      return loading;
    },
    get error() {
      return error;
    },
    get confirmed() {
      return confirmed;
    },
    get result() {
      return result;
    },
    get canConfirm() {
      return canConfirm;
    },
    get job() {
      return job;
    },
    get detached() {
      return detached;
    },
    get classId() {
      return classId;
    },
    set classId(v: number | null) {
      classId = v;
    },
    get reason() {
      return reason;
    },
    set reason(v: string) {
      reason = v;
    },
    get clusterUid() {
      return clusterUid;
    },
    set clusterUid(v: string | null) {
      clusterUid = v;
    },
    get clusterOptions() {
      return clusterOptions;
    },
    get limit() {
      return limit;
    },
    set limit(v: number | null) {
      limit = v;
    },
    get sample() {
      return sample;
    },
    set sample(v: SelectionSample | null) {
      sample = v;
    },
    get seed() {
      return seed;
    },
    set seed(v: number | null) {
      seed = v;
    },
    open,
    refresh,
    confirm,
    runInBackground,
    cancel,
  };
}

export type SelectionActionController = ReturnType<
  typeof createSelectionActionController
>;
