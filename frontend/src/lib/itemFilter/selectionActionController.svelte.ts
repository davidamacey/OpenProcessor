/**
 * "Act on every item a filter matches": exclude, restore, label as a
 * class, or move to a cluster, over an `ItemSelection` (the filter, plus
 * an optional limit / sample / seed) instead of a list of crop ids.
 *
 * The count the operator confirms is the server's: opening an action, or
 * changing the limit / sample / seed, runs the same write with
 * `dry_run: true` and shows its `selected`. Confirm repeats it with
 * `dry_run: false` against the filter as it was when the dialog opened.
 * The ids the write served go to `undoStore` so Z reverts it. Refusals
 * (both/neither of crop ids and filter, an empty filter without a limit,
 * too many items) are the server's own message.
 */
import {
  ApiError,
  batchExcludeSelection,
  batchUnexcludeSelection,
  bulkLabelSelection,
  moveSelectionToCluster,
} from '$lib/api';
import { undoStore } from '$stores/undo.svelte';
import { toastStore } from '$stores/toast.svelte';
import type {
  ItemFilter,
  ItemSelection,
  SelectionDryRun,
  SelectionSample,
} from '$lib/types_itemFilter';

export type SelectionActionId = 'exclude' | 'unexclude' | 'label' | 'move';

export interface SelectionActionResult {
  summary: string;
}

type WriteBody = {
  updated?: number;
  excluded?: number;
  unexcluded?: number;
  updated_ids: string[];
  conflicts?: unknown[];
};

function isDryRun(r: unknown): r is SelectionDryRun {
  return !!r && typeof r === 'object' && (r as { dry_run?: unknown }).dry_run === true;
}

export function createSelectionActionController(filter: () => ItemFilter) {
  let action = $state<SelectionActionId | null>(null);
  let snapshot: ItemFilter = {};
  let classId = $state<number | null>(null);
  let clusterId = $state<number | null>(null);
  let limit = $state<number | null>(null);
  let sample = $state<SelectionSample | null>(null);
  let seed = $state<number | null>(null);
  let selected = $state<number | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let confirmed = $state(false);
  let result = $state<SelectionActionResult | null>(null);
  let inflight: AbortController | null = null;

  function selection(): ItemSelection {
    const sel: ItemSelection = { filter: snapshot };
    if (limit != null) sel.limit = limit;
    if (sample != null) sel.sample = sample;
    if (sample === 'random' && seed != null) sel.seed = seed;
    return sel;
  }

  /** The missing field an action needs before it can ask the server. */
  function missingParam(): boolean {
    if (action === 'label') return classId == null;
    if (action === 'move') return clusterId == null;
    return false;
  }

  function call(dryRun: boolean, signal: AbortSignal): Promise<unknown> {
    const sel = selection();
    switch (action) {
      case 'exclude':
        return batchExcludeSelection(sel, 'ignore', dryRun, signal);
      case 'unexclude':
        return batchUnexcludeSelection(sel, dryRun, signal);
      case 'label':
        return bulkLabelSelection(sel, classId!, dryRun, signal);
      case 'move':
        return moveSelectionToCluster(sel, clusterId!, dryRun, signal);
      default:
        return Promise.reject(new Error('no action chosen'));
    }
  }

  /** A structured refusal (`{detail: {error, message}}`) is worded by its
   *  served `message`; a plain one by its `detail`. */
  function refusal(e: unknown): string {
    if (e instanceof ApiError) {
      const detail = (e.body as { detail?: unknown } | null)?.detail;
      const message =
        detail && typeof detail === 'object'
          ? (detail as { message?: unknown }).message
          : null;
      if (typeof message === 'string' && message) return message;
      return e.detail ?? e.message;
    }
    return (e as Error).message;
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
      error = refusal(e);
    } finally {
      if (inflight === ctrl) {
        loading = false;
        inflight = null;
      }
    }
  }

  async function open(next: SelectionActionId): Promise<void> {
    action = next;
    snapshot = structuredClone($state.snapshot(filter()));
    confirmed = false;
    result = null;
    await refresh();
  }

  function summarize(body: WriteBody): string {
    const n = body.updated ?? body.excluded ?? body.unexcluded ?? body.updated_ids.length;
    const conflicts = body.conflicts?.length ?? 0;
    const verb =
      action === 'exclude'
        ? 'Ignored'
        : action === 'unexclude'
          ? 'Restored'
          : action === 'label'
            ? 'Labeled'
            : 'Moved';
    return conflicts > 0 ? `${verb} ${n} (${conflicts} blocked)` : `${verb} ${n}`;
  }

  const canConfirm = $derived(
    action != null && selected != null && selected > 0 && !loading && !confirmed,
  );

  async function confirm(): Promise<void> {
    if (!canConfirm) return;
    inflight?.abort();
    const ctrl = new AbortController();
    inflight = ctrl;
    loading = true;
    error = null;
    try {
      const res = await call(false, ctrl.signal);
      if (isDryRun(res)) {
        error = 'The server answered a dry run; nothing was written.';
        return;
      }
      const body = res as WriteBody;
      if (action === 'exclude') undoStore.recordExclusion(body.updated_ids);
      else if (action === 'unexclude') undoStore.recordUnexclusion(body.updated_ids);
      else undoStore.recordWrites(body.updated_ids);
      confirmed = true;
      result = { summary: summarize(body) };
      toastStore.success(`${result.summary}.`);
    } catch (e) {
      error = refusal(e);
    } finally {
      if (inflight === ctrl) inflight = null;
      loading = false;
    }
  }

  function cancel(): void {
    inflight?.abort();
    inflight = null;
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
    get classId() {
      return classId;
    },
    set classId(v: number | null) {
      classId = v;
    },
    get clusterId() {
      return clusterId;
    },
    set clusterId(v: number | null) {
      clusterId = v;
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
    cancel,
  };
}

export type SelectionActionController = ReturnType<
  typeof createSelectionActionController
>;
