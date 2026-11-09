/*
 * Auto-label run controller: the job state, its poll loop (fast while the
 * server works on a run, slow otherwise) and the start / pause / resume /
 * cancel actions with their refusal-to-toast mapping. Call during
 * component initialisation (onMount/onDestroy).
 */
import { onDestroy, onMount } from 'svelte';
import {
  AUTO_LABEL_LIVE_STATUSES,
  cancelAutoLabel,
  getAutoLabelStatus,
  pauseAutoLabel,
  resumeAutoLabel,
  startAutoLabel,
  unknownStrategyDetail,
  type AutoLabelJobState,
  type AutoLabelStatus,
} from '$lib/api_autoLabel';
import { apiErrorText } from '$lib/api_errors';
import { configErrorDetail } from '$lib/api_profiles';
import type { AssistScope } from '$lib/assistScope.svelte';
import { resolveAutoLabelRunVlm } from '$lib/autoLabelRunVlm';
import { toastStore } from '$stores/toast.svelte';

const STAGE_ORDER = ['vlm', 'finalize'];

// The embed stage runs first, and only when the run asked for it (the
// job echoes its resolved args).
function stageOrderFor(j: AutoLabelJobState | null): string[] {
  return j?.args?.embed_missing === true || j?.stage === 'embed_missing'
    ? ['embed_missing', ...STAGE_ORDER]
    : STAGE_ORDER;
}

function stageIndex(order: string[], s: string): number {
  const i = order.indexOf(s);
  return i < 0 ? 0 : i;
}

export function createAutoLabelController(
  scope: Pick<AssistScope, 'toStartParams'>,
  scopeClassName: () => string | null,
) {
  let job: AutoLabelJobState | null = $state(null);
  let busy: boolean = $state(false);
  // p2 (2026-09-24 interactive pass): "Recluster now" used to fire on one
  // click with no confirmation — it's a dataset-wide, potentially
  // long-running pipeline run, so gate it behind an explicit confirm
  // step the way the dashboard's other one-shot actions already are.
  let confirmOpen: boolean = $state(false);
  let pollTimer: ReturnType<typeof setTimeout> | null = null;
  // G5: explicit control for pipeline.py's `run_vlm` (default False
  // server-side) — off by default here too, matching the backend. A
  // scoped run (AssistScopeBar picked a class/pack) always implies
  // run_vlm regardless of this checkbox, since a scoped run that skips
  // the VLM stage it claims to scope is a no-op (see start()).
  let runVlm: boolean = $state(false);
  // v0.4.0: embed the items without a vector first. Sent only when ticked.
  let embedMissing: boolean = $state(false);

  async function poll(): Promise<void> {
    try {
      job = await getAutoLabelStatus();
    } catch {
      /* network blip — keep last job, try again next tick */
    }
    schedule();
  }

  function schedule(): void {
    if (pollTimer) clearTimeout(pollTimer);
    // Fast poll while running, slow when idle. The slow poll catches
    // daemon-initiated runs without hammering the API.
    const ms = isLive ? 3000 : 30000;
    pollTimer = setTimeout(() => void poll(), ms);
  }

  function requestStart(): void {
    confirmOpen = true;
  }

  function closeConfirmStart(): void {
    if (busy) return;
    confirmOpen = false;
  }

  async function start(): Promise<void> {
    confirmOpen = false;
    busy = true;
    try {
      const sendRunVlm = resolveAutoLabelRunVlm(scope.toStartParams(), runVlm);
      job = await startAutoLabel({
        vlm_concurrency: 16,
        // 0 = process every unvalidated crop. Operators can ramp down
        // later if they want to time-box a run.
        max_vlm_crops: 0,
        // `{}` whenever nothing is scoped, so the composed URL stays
        // byte-identical to every request this panel has ever sent.
        ...scope.toStartParams(),
        // Omitted (not `false`) when off, so an unscoped/unchecked run's
        // URL stays byte-identical to every request this panel has ever
        // sent — the backend already defaults run_vlm to False.
        ...(sendRunVlm ? { run_vlm: true } : {}),
        ...(embedMissing ? { embed_missing: true } : {}),
      });
      toastStore.success(
        scopeClassName()
          ? `Auto-label queued — VLM labeling limited to ${scopeClassName()} (VLM stage will run).`
          : sendRunVlm
            ? 'Auto-label queued — running the VLM labeling stage.'
            : 'Auto-label queued.',
      );
      schedule();
    } catch (e) {
      const msg = apiErrorText(e);
      const refusal = configErrorDetail(e);
      if (unknownStrategyDetail(e)) {
        const d = unknownStrategyDetail(e)!;
        toastStore.error(
          `Start failed: unknown ${d.axis.replace('_', ' ')} "${d.requested}" — valid: ${d.valid_ids.join(', ') || 'none'}.`,
        );
      } else if (refusal) {
        // A served refusal (`vlm_external_not_acknowledged`,
        // `vlm_not_configured`, `vlm_endpoint_unavailable`, a pairing
        // `validation_failed`): its own message, never the generic 409.
        toastStore.error(`Start failed: ${refusal.message}`);
      } else if (msg.includes('409') || msg.includes('already in progress')) {
        toastStore.warn('An auto-label run is already queued, running or paused.');
        void poll();
      } else {
        toastStore.error(`Start failed: ${msg}`);
      }
    } finally {
      busy = false;
    }
  }

  async function cancel(): Promise<void> {
    busy = true;
    try {
      job = await cancelAutoLabel();
      toastStore.info('Cancel requested. Pipeline will stop at the next checkpoint.');
    } catch (e) {
      toastStore.error(`Cancel failed: ${apiErrorText(e)}`);
    } finally {
      busy = false;
    }
  }

  async function pause(): Promise<void> {
    busy = true;
    try {
      job = await pauseAutoLabel();
    } catch (e) {
      toastStore.error(`Pause failed: ${apiErrorText(e)}`);
    } finally {
      busy = false;
    }
  }

  async function resume(): Promise<void> {
    busy = true;
    try {
      job = await resumeAutoLabel();
      schedule();
    } catch (e) {
      toastStore.error(`Resume failed: ${apiErrorText(e)}`);
    } finally {
      busy = false;
    }
  }

  onMount(() => {
    void poll();
  });

  onDestroy(() => {
    if (pollTimer) clearTimeout(pollTimer);
  });

  // The server works on the run, or is stopping it (queued, running, cancelling).
  const isLive: boolean = $derived(
    AUTO_LABEL_LIVE_STATUSES.includes(
      ((job as AutoLabelJobState | null)?.status ?? 'idle') as AutoLabelStatus,
    ),
  );
  const isPaused: boolean = $derived(
    ((job as AutoLabelJobState | null)?.status ?? '') === 'paused',
  );
  // The run holds the project (live or paused): no new start until it ends.
  const hasRun: boolean = $derived(isLive || isPaused);
  const stageOrder: string[] = $derived(stageOrderFor(job as AutoLabelJobState | null));
  const stageIdx: number = $derived(
    stageIndex(stageOrder, (job as AutoLabelJobState | null)?.stage ?? ''),
  );
  const percent: number | null = $derived.by(() => {
    if (!job || job.total <= 0) return null;
    return Math.min(100, Math.round((job.processed / job.total) * 100));
  });

  const stageDurations: Array<[string, number]> = $derived.by(() => {
    const sd = (job as AutoLabelJobState | null)?.stage_durations;
    if (!sd) return [];
    return Object.entries(sd) as Array<[string, number]>;
  });
  const lastFinishedRel: string = $derived.by(() => {
    if (!job || !job.finished_at) return '';
    const seconds = Math.max(0, Date.now() / 1000 - job.finished_at);
    if (seconds < 60) return `${Math.round(seconds)}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  });

  return {
    get job() {
      return job;
    },
    get busy() {
      return busy;
    },
    get confirmOpen() {
      return confirmOpen;
    },
    get runVlm() {
      return runVlm;
    },
    set runVlm(v: boolean) {
      runVlm = v;
    },
    get embedMissing() {
      return embedMissing;
    },
    set embedMissing(v: boolean) {
      embedMissing = v;
    },
    get isLive() {
      return isLive;
    },
    get isPaused() {
      return isPaused;
    },
    get hasRun() {
      return hasRun;
    },
    get stageOrder() {
      return stageOrder;
    },
    get stageIdx() {
      return stageIdx;
    },
    get percent() {
      return percent;
    },
    get stageDurations() {
      return stageDurations;
    },
    get lastFinishedRel() {
      return lastFinishedRel;
    },
    requestStart,
    closeConfirmStart,
    start,
    cancel,
    pause,
    resume,
  };
}
