/**
 * StrategiesStore — Svelte 5 runes-based cache for the `/curation/methods`
 * capability-discovery response (curation-strategy plan, Phase 0).
 *
 * Read-only infrastructure: `init()` fetches once, the first time
 * anything calls it, and every later call is a no-op that returns the
 * same in-flight/resolved promise. Nothing in the UI wires this up yet
 * (Phase 3 wires the strategy bar to it) — this store exists purely so
 * later phases have a ready single source of truth for which cluster
 * methods / review sorts / overlays / scores the backend currently
 * offers.
 *
 * Unlike classesStore/healthStore there is no polling loop: capability
 * lists change on deploy, not per-session, and `getMethods()` already
 * degrades gracefully to `FALLBACK_METHODS` on any 404/network failure
 * (plan §5.3) — so a failed load isn't something a retry-on-a-timer would
 * fix. This store adds no window/document event listeners; it is purely
 * a data cache, not a UI concern.
 */

import { getMethods } from '$lib/api';
import { FALLBACK_METHODS, type OpMethodsResponse } from '$lib/strategies';

class StrategiesStore {
  methods = $state<OpMethodsResponse>(FALLBACK_METHODS);
  loading = $state<boolean>(false);
  loaded = $state<boolean>(false);
  error = $state<string | null>(null);

  /** Convenience accessor for the entry the backend (or the fallback)
   *  marks as its default cluster method — 'ivf' until a server ever
   *  says otherwise. Nothing writes cluster_id off this; it's read-only
   *  capability metadata. */
  defaultClusterMethodId = $derived(
    this.methods.cluster_methods.find((m) => m.default)?.id ?? 'ivf',
  );

  #inflight: Promise<void> | null = null;

  /**
   * Idempotent load. The first caller triggers the fetch; concurrent or
   * later callers (e.g. a second route mounting) reuse the same
   * promise/result rather than re-fetching. `getMethods()` is designed
   * to never reject (see `api.ts`), so `error` here only reflects an
   * unexpected throw from something other than the network path itself
   * — kept for future observability, not because it's expected to fire.
   */
  async init(): Promise<void> {
    if (this.loaded) return;
    if (this.#inflight) return this.#inflight;
    this.loading = true;
    this.#inflight = (async () => {
      try {
        this.methods = await getMethods();
        this.error = null;
      } catch (e) {
        if ((e as Error)?.name === 'AbortError') return;
        this.methods = FALLBACK_METHODS;
        this.error = (e as Error)?.message ?? 'failed to load /curation/methods';
      } finally {
        this.loading = false;
        this.loaded = true;
        this.#inflight = null;
      }
    })();
    return this.#inflight;
  }

  /** Drop the cache and re-fetch on the next `init()` call. Not wired to
   *  anything yet — provided for symmetry with classesStore's
   *  clearAndRefetch so Phase 3 doesn't need to touch this file again
   *  just to invalidate after a deploy. */
  reset(): void {
    this.methods = FALLBACK_METHODS;
    this.loaded = false;
    this.error = null;
  }
}

export const strategiesStore = new StrategiesStore();
