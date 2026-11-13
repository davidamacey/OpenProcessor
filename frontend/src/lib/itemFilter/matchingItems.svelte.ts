/**
 * The "Matching items" list on /clusters: `GET /crops` under the shared item
 * filter (including the open-vocabulary set / prompt, which only `/crops`
 * declares), paged by the same accumulating pager the other grids use.
 * The filter is read when each page is fetched, so every page carries the
 * filter in force when the list was (re)loaded.
 */
import { getCrops } from '$lib/api_cropRead';
import { createPager, type Pager } from '$lib/pager.svelte';
import type { Crop, HiddenCounts } from '$lib/types';
import type { ItemFilterQuery } from '$lib/types_itemFilter';

/** A page is a few screens of cards, so reading deep into a large project is not one network
 *  round trip per screen. `GET /crops` serves at most 500. */
export const MATCHING_PAGE_SIZE = 200;

export interface MatchingItems {
  readonly pager: Pager<Crop>;
  /** What the list hides, as the last page served it. */
  readonly hidden: HiddenCounts | null;
}

export function createMatchingItems(query: () => ItemFilterQuery): MatchingItems {
  let hidden = $state<HiddenCounts | null>(null);
  const pager = createPager<Crop>({
    fetchPage: async (cursor) => {
      const res = await getCrops({ ...query(), limit: MATCHING_PAGE_SIZE }, cursor);
      hidden = res.hidden ?? null;
      return res;
    },
    keyOf: (c) => c.id,
  });
  return {
    pager,
    get hidden() {
      return hidden;
    },
  };
}
