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

  /** Query params this tab's served entry honours; `null` when the
   *  endpoint is absent, not yet loaded, or didn't serve `filters` —
   *  callers treat `null` as "unknown, show every control" rather than
   *  hiding everything on a stale/older backend. */
  filtersFor(endpointId: string): string[] | null {
    return this.#byId.get(endpointId)?.filters ?? null;
  }

  /** Whether a given query param is in this tab's served `filters` list.
   *  Defaults to `true` (visible) when `filters` is unknown, so a
   *  missing/older `GET {API_PREFIX}/review/tabs` never hides a control that
   *  used to always render. */
  filterSupported(endpointId: string, param: string): boolean {
    const filters = this.filtersFor(endpointId);
    return filters == null || filters.includes(param);
  }

  /** This tab's served `filter_defaults[key]`, or `null` when absent. */
  filterDefault(endpointId: string, key: string): unknown | null {
    const v = this.#byId.get(endpointId)?.filter_defaults?.[key];
    return v === undefined ? null : v;
  }
}

export const reviewTabsVocabularyStore = new ReviewTabsVocabularyStore();
