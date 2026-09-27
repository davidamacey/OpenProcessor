/**
 * HealthStore — polls the GLOBAL {globalApi()}/health every 15s for the
 * top-bar OK/down chip, and the project-SCOPED {scoped()}/health
 * alongside it to keep `regionProfileStore` (and any other project
 * facts) fresh (P1 projects cutover — see CLAUDE.md "Health split").
 * The chip only ever reflects the global read: a project-scoped outage
 * (e.g. this project's OpenSearch indexes) shouldn't have to also poll
 * a second global-shaped source for "is the API up at all".
 *
 * Polling auto-stops when the tab is hidden (Page Visibility API) and
 * resumes on focus. A 404/503/network failure flips `ok` to false; the next
 * successful poll restores it.
 */

import { getGlobalHealth, getHealth } from '$lib/api';
import type { ApiHealth, GlobalHealth } from '$lib/types';
import { regionProfileStore } from '$stores/regionProfile.svelte';

const POLL_INTERVAL_MS = 15_000;

class HealthStore {
  health = $state<GlobalHealth | null>(null);
  scopedHealth = $state<ApiHealth | null>(null);
  ok = $state<boolean>(false);
  lastChecked = $state<number | null>(null);
  error = $state<string | null>(null);

  #timer: ReturnType<typeof setInterval> | null = null;
  #abort: AbortController | null = null;
  #refCount = 0;
  #stopHandle: ReturnType<typeof setTimeout> | null = null;

  acquire(): () => void {
    // m26 (2026-09-24 interactive pass): the root layout's mount/
    // release/remount during app boot (verified live: acquire() fires
    // 3x, refCount 0 each time — an SPA-boot quirk, not a bug in this
    // store) used to tear down and immediately restart polling each
    // time, aborting an in-flight /health request and refiring a new
    // one. A pending #stop() is now cancelled by a same-tick-or-soon
    // re-acquire instead of actually running.
    if (this.#stopHandle) {
      clearTimeout(this.#stopHandle);
      this.#stopHandle = null;
    }
    this.#refCount += 1;
    if (this.#refCount === 1 && !this.#timer) this.#start();
    return () => this.#release();
  }

  #release(): void {
    this.#refCount = Math.max(0, this.#refCount - 1);
    if (this.#refCount === 0 && !this.#stopHandle) {
      this.#stopHandle = setTimeout(() => {
        this.#stopHandle = null;
        if (this.#refCount === 0) this.#stop();
      }, 0);
    }
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
      const h = await getGlobalHealth(ctrl.signal);
      this.health = h;
      // Only 'down' or a network error surfaces the red banner: 'degraded'
      // means a non-critical dependency (e.g. the VLM) is intermittent
      // and labeling still works.
      this.ok = h?.status === 'ok' || h?.status === 'degraded';
      this.error = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.ok = false;
      this.error = (e as Error).message;
    } finally {
      this.lastChecked = Date.now();
    }
    // Project-scoped health, alongside the global read above — feeds
    // regionProfileStore/any project facts. A failure here is silent:
    // the scoped-health poll only ever refreshes facts the app already
    // has a fallback for (see regionProfileStore's own unknown/changed
    // handling), and shouldn't flip the global API-down chip.
    try {
      const sh = await getHealth(ctrl.signal);
      this.scopedHealth = sh;
      regionProfileStore.observe(sh?.region_profile);
    } catch {
      // ignore — the global poll above already recorded chip state
    }
  }
}

export type HealthChip = 'checking' | 'ok' | 'down';

/** Before the first poll settles there's no evidence either way, so the chip must not claim "down". */
export function healthChip(ok: boolean, lastChecked: number | null): HealthChip {
  if (lastChecked === null) return 'checking';
  return ok ? 'ok' : 'down';
}

export const HEALTH_CHIP_TEXT: Record<HealthChip, string> = {
  checking: 'API …',
  ok: 'API OK',
  down: 'API down',
};

export const healthStore = new HealthStore();
