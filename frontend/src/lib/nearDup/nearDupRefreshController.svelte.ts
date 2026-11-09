/**
 * Clears a `near_dup_stale` refusal: starts one run per stale level and
 * follows each job to its end. The refused action (freeze, export) is retried
 * by the user afterwards; this controller never retries it.
 */
import { apiErrorText } from '$lib/api_errors';
import { startNearDupRun } from '$lib/api_nearDup';
import { followJob } from '$lib/jobs/followJob';
import { runModeFor } from '$lib/nearDup/staleRefusal';
import type { JobView } from '$lib/types_jobs';
import type { NearDupStaleLevel } from '$lib/types_nearDup';

export function createNearDupRefreshController() {
  let stale = $state<NearDupStaleLevel[]>([]);
  let busy = $state<boolean>(false);
  let message = $state<string | null>(null);
  let error = $state<string | null>(null);
  let controller: AbortController | null = null;

  function show(levels: NearDupStaleLevel[]): void {
    stale = levels;
    message = null;
    error = null;
  }

  function clear(): void {
    stale = [];
    message = null;
    error = null;
  }

  async function refresh(): Promise<void> {
    if (busy || stale.length === 0) return;
    busy = true;
    error = null;
    const mine = new AbortController();
    controller = mine;
    try {
      const jobs: { level: string; id: string }[] = [];
      for (const lv of stale) {
        const started = await startNearDupRun(
          { level: lv.level, mode: runModeFor(lv) },
          mine.signal,
        );
        jobs.push({ level: lv.level, id: started.job_id });
      }
      for (const j of jobs) {
        const done: JobView = await followJob(j.id, {
          signal: mine.signal,
          onUpdate: (job) => {
            message = `${j.level} level: ${job.status}`;
          },
        });
        if (done.status !== 'completed') {
          error = `${j.level} level run ${done.status}${done.error_message ? `: ${done.error_message}` : ''}`;
          return;
        }
      }
      stale = [];
      message = 'Near-duplicate groups are up to date. Run the action again.';
    } catch (e) {
      if (!mine.signal.aborted) error = apiErrorText(e);
    } finally {
      busy = false;
      if (controller === mine) controller = null;
    }
  }

  function dispose(): void {
    controller?.abort();
  }

  return {
    get stale() {
      return stale;
    },
    get busy() {
      return busy;
    },
    get message() {
      return message;
    },
    get error() {
      return error;
    },
    show,
    clear,
    refresh,
    dispose,
  };
}

export type NearDupRefreshController = ReturnType<typeof createNearDupRefreshController>;
