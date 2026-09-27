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
 * a region slot's tab).
 *
 * On failure (or before `init()` resolves) the map is empty and every
 * caller falls back to the tab's own static `label`/no tooltip — a failed
 * read never breaks `/review`'s tab bar.
 */

import {
  getReviewTabs,
  type ReviewEmptyState,
  type ReviewFilterSpec,
  type ReviewTabVocabularyEntry,
} from '$lib/api';
import { onProjectChange } from '$lib/projectChange';

class ReviewTabsVocabularyStore {
  list = $state<ReviewTabVocabularyEntry[]>([]);
  loaded = $state<boolean>(false);
  /** #36 item 9: whether this deployment has any probe predictions/item
   *  scores at all — `null` until loaded (or when the read failed). */
  emptyState = $state<ReviewEmptyState | null>(null);
  #byId = $derived(new Map(this.list.map((t) => [t.id, t])));
  #inflight: Promise<void> | null = null;
  #gen = 0;

  async init(): Promise<void> {
    if (this.loaded) return;
    if (this.#inflight) return this.#inflight;
    const gen = this.#gen;
    this.#inflight = (async () => {
      try {
        const res = await getReviewTabs();
        // A load started for the previous project never lands.
        if (gen !== this.#gen) return;
        this.list = res.tabs;
        this.emptyState = res.empty_state;
      } catch {
        if (gen !== this.#gen) return;
        this.list = [];
      } finally {
        if (gen === this.#gen) {
          this.loaded = true;
          this.#inflight = null;
        }
      }
    })();
    return this.#inflight;
  }

  /** Served label for a tab/preset endpoint id; `fallback` (the tab's own
   *  static label) when not loaded or the served list doesn't know this id. */
  labelFor(endpointId: string, fallback: string): string {
    return this.#byId.get(endpointId)?.label ?? fallback;
  }

  /** Served description, for use as a tooltip; `null` when empty or unknown. */
  descriptionFor(endpointId: string): string | null {
    return this.#byId.get(endpointId)?.description || null;
  }

  /** Query params this tab's served entry honours; `null` when not yet
   *  loaded or the served list has no entry for this id (e.g. a tier-2
   *  slot tab the backend doesn't know) — callers treat `null` as
   *  "unknown, show every control". */
  filtersFor(endpointId: string): string[] | null {
    return this.#byId.get(endpointId)?.filters ?? null;
  }

  /** Whether a given query param is in this tab's served `filters` list.
   *  Defaults to `true` (visible) when the tab has no served entry. */
  filterSupported(endpointId: string, param: string): boolean {
    const filters = this.filtersFor(endpointId);
    return filters == null || filters.includes(param);
  }

  /** This tab's served `filter_defaults[key]`, or `null` when absent. */
  filterDefault(endpointId: string, key: string): unknown | null {
    const v = this.#byId.get(endpointId)?.filter_defaults[key];
    return v === undefined ? null : v;
  }

  /** This tab's served self-describing enum filters (3f1a11e adoption) —
   *  empty array when not yet loaded, unknown, or the tab declares none. Drives the generic served-enum filter bar: a future
   *  spec on any tab renders with zero page-specific code, since the
   *  page never reads a `param` by name. */
  filterSpecsFor(endpointId: string): ReviewFilterSpec[] {
    return this.#byId.get(endpointId)?.filter_specs ?? [];
  }

  /** Project switch: the vocabulary is per project. */
  resetForProjectChange(): void {
    this.#gen += 1;
    this.#inflight = null;
    this.list = [];
    this.emptyState = null;
    this.loaded = false;
  }
}

export const reviewTabsVocabularyStore = new ReviewTabsVocabularyStore();
onProjectChange(() => reviewTabsVocabularyStore.resetForProjectChange());
