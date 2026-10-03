/**
 * The "Matching items" list on /clusters: `GET /crops` under the shared item
 * filter (including the open-vocabulary set / prompt, which only `/crops`
 * declares), paged by the same accumulating pager the other grids use.
 * The filter is read when each page is fetched, so every page carries the
 * filter in force when the list was (re)loaded.
 */
import { getCrops } from '$lib/api';
import { createPager, type Pager } from '$lib/pager.svelte';
import type { Crop } from '$lib/types';
import type { ItemFilterQuery } from '$lib/types_itemFilter';

export const MATCHING_PAGE_SIZE = 48;

export interface MatchingItems {
  readonly pager: Pager<Crop>;
}

export function createMatchingItems(query: () => ItemFilterQuery): MatchingItems {
  const pager = createPager<Crop>({
    fetchPage: async (page) => getCrops({ ...query(), page, limit: MATCHING_PAGE_SIZE }),
    keyOf: (c) => c.id,
  });
  return { pager };
}
