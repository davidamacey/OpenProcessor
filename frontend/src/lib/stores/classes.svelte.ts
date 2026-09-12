/**
 * ClassesStore — Svelte 5 runes-based class registry cache.
 *
 * - Auto-refreshes every 30s while there is an active subscriber.
 * - Survives route changes (singleton).
 * - Top-N-for-cluster is computed client-side from currently loaded crops if
 *   the server doesn't provide an authoritative answer.
 */

import { getClasses } from '$lib/api';
import { isAssignableClass } from '$lib/classVisibility';
import type { OpClass } from '$lib/types';

const REFRESH_INTERVAL_MS = 30_000;

class ClassesStore {
  classes = $state<OpClass[]>([]);
  loading = $state<boolean>(false);
  lastUpdated = $state<number | null>(null);
  error = $state<string | null>(null);

  #timer: ReturnType<typeof setInterval> | null = null;
  #abort: AbortController | null = null;
  #refCount = 0;

  /** Increment subscriber count; first subscriber starts polling. */
  acquire(): () => void {
    this.#refCount += 1;
    if (this.#refCount === 1) {
      void this.refresh();
      this.#timer = setInterval(() => void this.refresh(), REFRESH_INTERVAL_MS);
    }
    return () => this.#release();
  }

  #release(): void {
    this.#refCount = Math.max(0, this.#refCount - 1);
    if (this.#refCount === 0) {
      if (this.#timer) clearInterval(this.#timer);
      this.#timer = null;
      this.#abort?.abort();
      this.#abort = null;
    }
  }

  async refresh(): Promise<void> {
    this.#abort?.abort();
    const ctrl = new AbortController();
    this.#abort = ctrl;
    this.loading = true;
    try {
      const data = await getClasses(ctrl.signal);
      this.classes = Array.isArray(data) ? data : [];
      this.lastUpdated = Date.now();
      this.error = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.error = (e as Error).message;
    } finally {
      this.loading = false;
    }
  }

  /**
   * Drop the current cache and re-fetch. Used after add/rename/merge so the
   * UI never displays a stale row briefly. The 30-second poll would catch
   * up on its own; this just makes mutations visible immediately.
   */
  async clearAndRefetch(): Promise<void> {
    this.classes = [];
    this.lastUpdated = null;
    await this.refresh();
  }

  byId(id: number): OpClass | undefined {
    return this.classes.find((c) => c.id === id);
  }

  byName(name: string): OpClass | undefined {
    const n = name.toLowerCase();
    return this.classes.find((c) => c.name.toLowerCase() === n);
  }

  /**
   * Returns the top-N most-frequent classes for a given cluster.
   * The server endpoint to do this exactly doesn't exist in MVP; instead we
   * fall back to the global most-frequent classes. Page-level code can
   * override by passing in a precomputed list.
   */
  topNForCluster(_clusterId: number, n = 10): OpClass[] {
    return this.classes
      .filter(isAssignableClass)
      .sort((a, b) => (b.validated_count ?? 0) - (a.validated_count ?? 0))
      .slice(0, n);
  }
}

export const classesStore = new ClassesStore();
