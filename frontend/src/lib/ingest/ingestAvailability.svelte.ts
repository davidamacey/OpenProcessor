/**
 * ingestAvailability — one-shot capability gate for `/ingest`, modelled
 * exactly on `src/lib/bakeoffAvailability.svelte.ts`
 * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.2).
 *
 * Probes `GET {API_PREFIX}/ingest/status`: an idempotent **read** whose
 * two possible outcomes are unambiguous — a real payload
 * (`{total, by_source, by_day}`) means "the ingest router is mounted",
 * and a 404/501 unambiguously means "it isn't." There is no conflation
 * between "unsupported" and "nothing ingested yet" the way probing a
 * write route would produce.
 *
 * BA-2 (`GET {API_PREFIX}/ingest/config`) has since landed
 * (OpenProcessor c676d2b), but this store still probes `/ingest/status`
 * deliberately — it stays the single, minimal availability signal;
 * `/routes/ingest/+page.svelte` fetches the richer `IngestConfig` itself
 * once `available` is confirmed `true`, so a config fetch never fires
 * against a backend that lacks the router at all.
 *
 * `available` starts `null` (optimistic) so the nav link renders
 * immediately and disappears only on *confirmed* absence, exactly like
 * `bakeoffAvailability` — see that module's doc comment for the full
 * rationale.
 */

import { ApiError, getIngestStatus } from '$lib/api';

class IngestAvailabilityStore {
  available = $state<boolean | null>(null);

  #loaded = false;
  #inflight: Promise<void> | null = null;

  async init(): Promise<void> {
    if (this.#loaded) return;
    if (this.#inflight) return this.#inflight;
    this.#inflight = (async () => {
      try {
        await getIngestStatus();
        this.available = true;
      } catch (e) {
        if (e instanceof ApiError && (e.status === 404 || e.status === 501)) {
          this.available = false;
        }
        // Any other error (network, 5xx, abort): leave `available`
        // unchanged — a transient outage must not hide a route that
        // actually exists.
      } finally {
        this.#loaded = true;
        this.#inflight = null;
      }
    })();
    return this.#inflight;
  }
}

export const ingestAvailability = new IngestAvailabilityStore();
