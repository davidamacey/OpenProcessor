/**
 * "Accept confident labels": the threshold and the sources, the served dry run (total, by class,
 * what is held back) and the one recorded operation that writes it. Detector confidence is about
 * the detector's own class, so this is a person's decision, never a setting: the dry run is
 * always shown first and the write toasts with Undo. When an accuracy audit exists, each class
 * row also carries its audited precision and interval (served by `GET /audit/report`).
 */
import { acceptMachine } from '$lib/api_labelOps';
import { getAuditReport } from '$lib/api_labelConfirmation';
import { apiErrorText } from '$lib/api_errors';
import { runLabelOp } from '$lib/labelOps/runLabelOp';
import type { AuditClassStat } from '$lib/types_labelConfirmation';
import type { ItemFilter } from '$lib/types_itemFilter';
import type { AcceptMachineResponse } from '$lib/types_labelOps';
import type { JobView } from '$lib/types_jobs';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';

export type AcceptSource = 'detector' | 'vlm';
export const DEFAULT_THRESHOLD = 0.95;

export function createAcceptConfident(filter: () => ItemFilter) {
  let threshold = $state(DEFAULT_THRESHOLD);
  let sources = $state<AcceptSource[]>(['detector', 'vlm']);
  let dryRun = $state<AcceptMachineResponse | null>(null);
  let precision = $state<Record<string, AuditClassStat>>({});
  let loading = $state(false);
  let applying = $state(false);
  let job = $state<JobView | null>(null);
  let error = $state<string | null>(null);
  let inflight: AbortController | null = null;

  const valid = $derived(threshold >= 0.5 && threshold <= 1 && sources.length > 0);
  const canApply = $derived(
    valid && !loading && !applying && dryRun != null && dryRun.items > 0,
  );

  function body(dry: boolean) {
    return {
      dry_run: dry,
      min_confidence: threshold,
      sources: [...sources],
      filter: filter(),
    };
  }

  async function refresh(): Promise<void> {
    inflight?.abort();
    if (!valid) {
      dryRun = null;
      return;
    }
    const ctrl = new AbortController();
    inflight = ctrl;
    loading = true;
    error = null;
    try {
      const res = await acceptMachine(body(true), ctrl.signal);
      if (!ctrl.signal.aborted) dryRun = res;
    } catch (e) {
      if (!ctrl.signal.aborted) error = apiErrorText(e);
    } finally {
      if (inflight === ctrl) {
        loading = false;
        inflight = null;
      }
    }
  }

  async function loadPrecision(): Promise<void> {
    try {
      const report = await getAuditReport();
      precision = Object.fromEntries(
        report.detector.filter((s) => !s.insufficient_sample).map((s) => [s.name, s]),
      );
    } catch {
      // No accuracy audit yet (or it cannot be read): the rows carry no precision.
      precision = {};
    }
  }

  async function open(): Promise<void> {
    await Promise.all([refresh(), loadPrecision()]);
  }

  function setThreshold(v: number): void {
    threshold = v;
    void refresh();
  }

  function toggleSource(s: AcceptSource): void {
    sources = sources.includes(s) ? sources.filter((x) => x !== s) : [...sources, s];
    void refresh();
  }

  /** Writes the accepted labels; resolves true once the operation has landed. */
  async function apply(): Promise<boolean> {
    if (!canApply) return false;
    applying = true;
    error = null;
    job = null;
    try {
      const res = await acceptMachine(body(false));
      const out = await runLabelOp(
        { op_id: res.op_id, job_id: res.job_id, status: 'queued' },
        { onUpdate: (j) => (job = j) },
      );
      undoStore.recordLabelOp(out.opId, []);
      toastStore.push({
        kind: 'success',
        text: `Accepted ${out.updated.toLocaleString()} labels.`,
        ttl_ms: 8000,
        action: { label: 'Undo', run: () => void undoStore.undoLast() },
      });
      return true;
    } catch (e) {
      error = apiErrorText(e);
      return false;
    } finally {
      applying = false;
    }
  }

  function dispose(): void {
    inflight?.abort();
  }

  return {
    get threshold() {
      return threshold;
    },
    get sources() {
      return sources;
    },
    get dryRun() {
      return dryRun;
    },
    get precision() {
      return precision;
    },
    get loading() {
      return loading;
    },
    get applying() {
      return applying;
    },
    get job() {
      return job;
    },
    get error() {
      return error;
    },
    get valid() {
      return valid;
    },
    get canApply() {
      return canApply;
    },
    open,
    setThreshold,
    toggleSource,
    apply,
    dispose,
  };
}

export type AcceptConfident = ReturnType<typeof createAcceptConfident>;
