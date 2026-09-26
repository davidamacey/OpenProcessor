/**
 * bakeoffAvailability — provisional capability gate for `/bakeoff`
 * (docs/design/bakeoff-train-genericization-plan-2026-09-21.md §2.5/§6
 * commit 1).
 *
 * `/bakeoff` was the only remaining backend-optional surface in the app
 * with no availability gate at all: the nav link rendered unconditionally
 * and `onMount` fired four GETs against a router
 * (`OpenProcessor`'s `bakeoff.py`) that may not be mounted on a given
 * Cropwright backend. This store closes that gap with a one-shot,
 * read-only probe.
 *
 * Why probing is acceptable here but was rejected for `/export`
 * (`isDatasetExportAvailable` in `$lib/strategies`): `GET
 * {API_PREFIX}/bakeoff/runs` is an idempotent **read**, and its two
 * possible outcomes are unambiguous — `{runs: []}` means "no runs yet,
 * route exists" and a 404/501 means "router not mounted." There is no
 * conflation between "unsupported" and "hasn't run yet" the way probing
 * `POST /export/{kind}` (a write) or its `/status` sibling would produce.
 *
 * This whole module is **provisional**. The correct long-term shape is a
 * `GET {API_PREFIX}/methods` `evaluation` axis (see this plan's §8 — a
 * cross-repo ask, not landed) with an `isEvaluationAvailable()` gate
 * mirroring `isDatasetExportAvailable` exactly. The moment that axis
 * ships, delete this module and point both call sites
 * (`src/routes/+layout.svelte`, `src/routes/bakeoff/+page.svelte`) at
 * `isEvaluationAvailable(strategiesStore.methods.evaluations, 'bakeoff')`
 * instead.
 *
 * The nav link renders optimistically (`available` starts `null`, and
 * `!== false` is the render condition) and hides only on *confirmed*
 * absence, so a backend without the route shows the link briefly and then
 * removes it once, on first paint after the probe resolves. That is
 * strictly better than today's permanently-broken link and disappears
 * entirely once the `/methods` axis lands. Do not "fix" the flicker by
 * making the link render-blocking.
 */

import { ApiError, bakeoffRuns } from '$lib/api';

class BakeoffAvailabilityStore {
  /** `null` = not yet determined (optimistic — render as available). */
  available = $state<boolean | null>(null);

  #loaded = false;
  #inflight: Promise<void> | null = null;

  /**
   * Idempotent, never-rejecting load — same shape as
   * `strategiesStore.init()`. On success, `available = true`. On a 404 or
   * 501 (router not mounted), `available = false`. On any other failure
   * (network error, 5xx, abort), `available` is left at its current,
   * optimistic value: a transient outage must not hide a route that
   * actually exists.
   */
  async init(): Promise<void> {
    if (this.#loaded) return;
    if (this.#inflight) return this.#inflight;
    this.#inflight = (async () => {
      try {
        await bakeoffRuns();
        this.available = true;
      } catch (e) {
        if (e instanceof ApiError && (e.status === 404 || e.status === 501)) {
          this.available = false;
        }
        // Any other error: leave `available` unchanged (fail open).
      } finally {
        this.#loaded = true;
        this.#inflight = null;
      }
    })();
    return this.#inflight;
  }
}

export const bakeoffAvailability = new BakeoffAvailabilityStore();
