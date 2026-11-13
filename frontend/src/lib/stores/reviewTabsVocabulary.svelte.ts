/**
 * ReviewTabsVocabularyStore — the served review queues (`GET {API_PREFIX}/review/tabs`): their
 * label, description, honoured filters and, under the filter the rail loaded them with, their
 * count. Keyed by each queue's id, which is also the `GET {API_PREFIX}/review/{id}` segment (a
 * slot reads the queue named by its `QueueCapability.endpointId`, e.g. `'regions'`).
 *
 * On failure (or before the first load resolves) the list is empty and every caller falls back to
 * its own static label — a failed read never breaks the rail.
 */

import {
  getReviewTabs,
  type ReviewEmptyState,
  type ReviewFilterSpec,
  type ReviewTabVocabularyEntry,
} from '$lib/api_review';
import { onProjectChange } from '$lib/projectChange';

class ReviewTabsVocabularyStore {
  list = $state<ReviewTabVocabularyEntry[]>([]);
  loaded = $state<boolean>(false);
  /** #36 item 9: whether this deployment has any probe predictions/item
   *  scores at all — `null` until loaded (or when the read failed). */
  emptyState = $state<ReviewEmptyState | null>(null);
  /** The first queue with work, as served (`default_tab`); null when every queue is empty. */
  defaultTab = $state<string | null>(null);
  #byId = $derived(new Map(this.list.map((t) => [t.id, t])));
  #inflight: Promise<void> | null = null;
  #gen = 0;
  /** The filter the list was (or is being) loaded under. */
  #key: string | null = null;
  #seq = 0;

  /** Loads the vocabulary once, under no filter. */
  async init(): Promise<void> {
    return this.load();
  }

  /** Loads the queues with their counts under `filter`; the same filter again is a no-op. Only the
   *  latest request lands, and a load started for the previous project never does. */
  async load(filter: Record<string, unknown> = {}): Promise<void> {
    const key = JSON.stringify(filter);
    if (key === this.#key) return this.#inflight ?? undefined;
    this.#key = key;
    const gen = this.#gen;
    const seq = ++this.#seq;
    const current = (): boolean => gen === this.#gen && seq === this.#seq;
    const run = (async () => {
      try {
        const res = await getReviewTabs(filter);
        if (!current()) return;
        this.list = res.tabs;
        this.emptyState = res.empty_state;
        this.defaultTab = res.default_tab ?? null;
      } catch {
        if (!current()) return;
        this.list = [];
      } finally {
        if (current()) {
          this.loaded = true;
          this.#inflight = null;
        }
      }
    })();
    this.#inflight = run;
    return run;
  }

  /** Whether the served vocabulary has an entry for this endpoint id. */
  hasEntry(endpointId: string): boolean {
    return this.#byId.has(endpointId);
  }

  /** Served label for a tab/preset endpoint id; `fallback` (the tab's own
   *  static label) when not loaded or the served list doesn't know this id. */
  labelFor(endpointId: string, fallback: string): string {
    return this.#byId.get(endpointId)?.label ?? fallback;
  }

  /** Served item count of a tab; `null` when the served list has no entry for it. */
  countFor(endpointId: string): number | null {
    return this.#byId.get(endpointId)?.count ?? null;
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
    this.#key = null;
    this.list = [];
    this.emptyState = null;
    this.defaultTab = null;
    this.loaded = false;
  }
}

export const reviewTabsVocabularyStore = new ReviewTabsVocabularyStore();
onProjectChange(() => reviewTabsVocabularyStore.resetForProjectChange());
