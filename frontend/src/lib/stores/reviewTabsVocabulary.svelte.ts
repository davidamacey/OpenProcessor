/**
 * ReviewTabsVocabularyStore — served review-tab labels/descriptions
 * (`GET {API_PREFIX}/review/tabs`, W0 naming-sweep finding m9), loaded once.
 *
 * The frontend keeps owning its own tab *structure* and ids
 * (`reviewTabs.ts`'s `REVIEW_TABS`/`REVIEW_PRESETS`) — this store only
 * overlays the served `label`/`description` on top, keyed by each served
 * entry's `id`, which for every tab this deployment renders equals the
 * `endpointId`/preset id already used to call `GET {API_PREFIX}/review/{id}`
 * (core tabs and presets: `id === endpointId`; slot tabs: keyed by the
 * slot's `QueueCapability.endpointId`, e.g. `'regions'` for
 * `license_plate`'s Plates tab).
 *
 * On failure (or before `init()` resolves) the map is empty and every
 * caller falls back to the tab's own static `label`/no tooltip — a
 * missing endpoint never breaks `/review`'s tab bar.
 */

import { getReviewTabsVocabulary, type ReviewTabVocabularyEntry } from '$lib/api';

class ReviewTabsVocabularyStore {
  list = $state<ReviewTabVocabularyEntry[]>([]);
  loaded = $state<boolean>(false);
  #byId = $derived(new Map(this.list.map((t) => [t.id, t])));
  #inflight: Promise<void> | null = null;

  async init(): Promise<void> {
    if (this.loaded) return;
    if (this.#inflight) return this.#inflight;
    this.#inflight = (async () => {
      try {
        this.list = await getReviewTabsVocabulary();
      } catch {
        this.list = [];
      } finally {
        this.loaded = true;
        this.#inflight = null;
      }
    })();
    return this.#inflight;
  }

  /** Served label for a tab/preset endpoint id; `fallback` (the tab's own
   *  static label) when the endpoint is absent or doesn't know this id. */
  labelFor(endpointId: string, fallback: string): string {
    return this.#byId.get(endpointId)?.label ?? fallback;
  }

  /** Served description, for use as a tooltip; `null` when absent. */
  descriptionFor(endpointId: string): string | null {
    return this.#byId.get(endpointId)?.description ?? null;
  }
}

export const reviewTabsVocabularyStore = new ReviewTabsVocabularyStore();
