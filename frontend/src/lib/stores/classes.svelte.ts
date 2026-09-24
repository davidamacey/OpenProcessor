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
import type { ClassThresholds, RegistryClass } from '$lib/types';

const REFRESH_INTERVAL_MS = 30_000;

const EMPTY_THRESHOLDS: ClassThresholds = {
  block_below: 0,
  warn_below: 0,
  min_test_per_class: 0,
  aug_target_min: 0,
  aug_target_max: 0,
};

class ClassesStore {
  classes = $state<RegistryClass[]>([]);
  /** Server-computed adequacy/aug-target/test-minimum thresholds from
   *  `GET {API_PREFIX}/classes` — see `RegistryClass.adequacy`. Zeros until
   *  the first successful fetch. */
  thresholds = $state<ClassThresholds>(EMPTY_THRESHOLDS);
  /** Every single-character hotkey combo the server reserves for a
   *  labeling action — `classHotkey.ts` unions this with any registered
   *  slot's own keymap. Empty until the first successful fetch. */
  reservedHotkeys = $state<string[]>([]);
  loading = $state<boolean>(false);
  lastUpdated = $state<number | null>(null);
  error = $state<string | null>(null);

  #timer: ReturnType<typeof setInterval> | null = null;
  #abort: AbortController | null = null;
  #refCount = 0;
  #stopHandle: ReturnType<typeof setTimeout> | null = null;

  /** Increment subscriber count; first subscriber starts polling.
   *
   * m26 (2026-09-24 interactive pass): the root layout's mount/release/
   * remount during app boot (verified live: acquire() fires repeatedly
   * with refCount back at 0 each time — an SPA-boot quirk, not a bug in
   * this store) used to tear down and immediately restart polling each
   * time, aborting an in-flight fetch and refiring a new one. A pending
   * teardown is now cancelled by a same-tick-or-soon re-acquire instead
   * of actually running. */
  acquire(): () => void {
    if (this.#stopHandle) {
      clearTimeout(this.#stopHandle);
      this.#stopHandle = null;
    }
    this.#refCount += 1;
    if (this.#refCount === 1 && !this.#timer) {
      void this.refresh();
      this.#timer = setInterval(() => void this.refresh(), REFRESH_INTERVAL_MS);
    }
    return () => this.#release();
  }

  #release(): void {
    this.#refCount = Math.max(0, this.#refCount - 1);
    if (this.#refCount === 0 && !this.#stopHandle) {
      this.#stopHandle = setTimeout(() => {
        this.#stopHandle = null;
        if (this.#refCount !== 0) return;
        if (this.#timer) clearInterval(this.#timer);
        this.#timer = null;
        this.#abort?.abort();
        this.#abort = null;
      }, 0);
    }
  }

  async refresh(): Promise<void> {
    this.#abort?.abort();
    const ctrl = new AbortController();
    this.#abort = ctrl;
    this.loading = true;
    try {
      const data = await getClasses(ctrl.signal);
      this.classes = data.classes;
      this.thresholds = data.thresholds;
      this.reservedHotkeys = data.reserved_hotkeys;
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

  byId(id: number): RegistryClass | undefined {
    return this.classes.find((c) => c.id === id);
  }

  byName(name: string): RegistryClass | undefined {
    const n = name.toLowerCase();
    return this.classes.find((c) => c.name.toLowerCase() === n);
  }

  /**
   * Returns the top-N most-frequent classes for a given cluster.
   * The server endpoint to do this exactly doesn't exist in MVP; instead we
   * fall back to the global most-frequent classes. Page-level code can
   * override by passing in a precomputed list.
   */
  topNForCluster(_clusterId: number, n = 10): RegistryClass[] {
    return this.classes
      .filter(isAssignableClass)
      .sort((a, b) => (b.validated_count ?? 0) - (a.validated_count ?? 0))
      .slice(0, n);
  }
}

export const classesStore = new ClassesStore();
