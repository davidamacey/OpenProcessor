/**
 * "Pick a diverse sample": starts the served selection job over the Workspace filter, follows it,
 * and hands the picked ids to the Workspace as its selection. The server decides whether the pool
 * is small enough to answer at once; a large pool is a job this polls (`GET /select/status`) and
 * whose picked items it then reads a page at a time (`GET /select/diverse/{job_id}/items`).
 */
import { apiErrorText } from '$lib/api_errors';
import {
  cancelSelect,
  getDiverseItems,
  getSelectStatus,
  selectDiverse,
} from '$lib/api_select';
import type { ItemFilter } from '$lib/types_itemFilter';

const POLL_MS = 2000;
const DEFAULT_K = 100;

export type DiversePhase = 'idle' | 'running';

interface Host {
  itemBody(): ItemFilter;
  selectItems(ids: string[]): void;
}

export function createDiverseSample(host: Host) {
  let k = $state(DEFAULT_K);
  let phase = $state<DiversePhase>('idle');
  let message = $state<string | null>(null);
  let timer: ReturnType<typeof setTimeout> | null = null;
  // A cancel or a new start bumps this, so a poll that was already in flight cannot select.
  let run = 0;

  function stopPolling(): void {
    if (timer) clearTimeout(timer);
    timer = null;
  }

  function done(ids: string[], nPool: number): void {
    phase = 'idle';
    host.selectItems(ids);
    message = `Picked ${ids.length} of ${nPool}`;
  }

  /** The picked ids of a finished job, read page by page; null when the run was superseded. */
  async function readPicked(jobId: string, mine: number): Promise<string[] | null> {
    const ids: string[] = [];
    let cursor: string | null = null;
    do {
      const page = await getDiverseItems(jobId, cursor);
      if (mine !== run) return null;
      ids.push(...page.ids);
      cursor = page.next_cursor;
    } while (cursor);
    return ids;
  }

  async function poll(mine: number): Promise<void> {
    try {
      const st = await getSelectStatus();
      if (mine !== run) return;
      if (st.status === 'completed' && st.result && st.job_id) {
        const ids = await readPicked(st.job_id, mine);
        if (ids) done(ids, st.result.n_pool);
        return;
      }
      if (st.status === 'failed' || st.status === 'cancelled' || st.status === 'idle') {
        phase = 'idle';
        message =
          st.error ??
          `The selection ${st.status === 'idle' ? 'did not run' : st.status}.`;
        return;
      }
      timer = setTimeout(() => void poll(mine), POLL_MS);
    } catch (e) {
      if (mine !== run) return;
      phase = 'idle';
      message = apiErrorText(e);
    }
  }

  async function start(): Promise<void> {
    stopPolling();
    const mine = ++run;
    message = null;
    phase = 'running';
    try {
      const res = await selectDiverse({ item_filter: host.itemBody() }, k);
      if (mine !== run) return;
      if (res.kind === 'ready') return done(res.selection.crop_ids, res.selection.n_pool);
      if (res.kind === 'job') {
        timer = setTimeout(() => void poll(mine), POLL_MS);
        return;
      }
      phase = 'idle';
      message =
        res.kind === 'disabled'
          ? 'Diverse selection is turned off on this server.'
          : 'A selection is already running; wait for it or cancel it.';
    } catch (e) {
      if (mine !== run) return;
      phase = 'idle';
      message = apiErrorText(e);
    }
  }

  async function cancel(): Promise<void> {
    run++;
    stopPolling();
    phase = 'idle';
    try {
      await cancelSelect();
      message = 'Selection cancelled.';
    } catch (e) {
      message = apiErrorText(e);
    }
  }

  return {
    get k() {
      return k;
    },
    set k(v: number) {
      k = v;
    },
    get phase() {
      return phase;
    },
    get message() {
      return message;
    },
    start,
    cancel,
    dispose: () => {
      run++;
      stopPolling();
    },
  };
}

export type DiverseSample = ReturnType<typeof createDiverseSample>;
