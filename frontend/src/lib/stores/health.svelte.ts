/**
 * HealthStore — polls /curation/health every 15s and exposes an OK/down indicator.
 *
 * Polling auto-stops when the tab is hidden (Page Visibility API) and
 * resumes on focus. A 404/503/network failure flips `ok` to false; the next
 * successful poll restores it.
 */

import { getHealth } from '$lib/api';
import type { OpHealth } from '$lib/types';

const POLL_INTERVAL_MS = 15_000;

class HealthStore {
  health = $state<OpHealth | null>(null);
  ok = $state<boolean>(false);
  lastChecked = $state<number | null>(null);
  error = $state<string | null>(null);

  #timer: ReturnType<typeof setInterval> | null = null;
  #abort: AbortController | null = null;
  #refCount = 0;

  acquire(): () => void {
    this.#refCount += 1;
    if (this.#refCount === 1) this.#start();
    return () => this.#release();
  }

  #release(): void {
    this.#refCount = Math.max(0, this.#refCount - 1);
    if (this.#refCount === 0) this.#stop();
  }

  #start(): void {
    if (typeof window === 'undefined') return;
    void this.poll();
    this.#timer = setInterval(() => void this.poll(), POLL_INTERVAL_MS);
    document.addEventListener('visibilitychange', this.#onVis);
  }

  #stop(): void {
    if (this.#timer) clearInterval(this.#timer);
    this.#timer = null;
    this.#abort?.abort();
    this.#abort = null;
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this.#onVis);
    }
  }

  #onVis = (): void => {
    if (document.visibilityState === 'visible') void this.poll();
  };

  async poll(): Promise<void> {
    this.#abort?.abort();
    const ctrl = new AbortController();
    this.#abort = ctrl;
    try {
      const h = (await getHealth(ctrl.signal)) as OpHealth & {
        status?: string;
      };
      this.health = h;
      // /curation/health returns {status: 'ok'|'degraded'|'down', triton, ...},
      // while older /health returned {ok: bool, components}. Treat both
      // 'ok' and 'degraded' as up — only 'down' or a network error
      // should surface the red banner. Degraded means a non-critical
      // dependency (e.g. Gemma) is intermittent; the labeler still works.
      // Old /health: {ok: bool}. New /curation/health: {status: 'ok'|'degraded'|'down'}.
      // 'degraded' (e.g. Gemma unavailable) still lets labeling work fine.
      const status = h?.status;
      this.ok = h?.ok !== undefined ? !!h.ok : status === 'ok' || status === 'degraded';
      this.error = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.ok = false;
      this.error = (e as Error).message;
    } finally {
      this.lastChecked = Date.now();
    }
  }
}

export const healthStore = new HealthStore();
