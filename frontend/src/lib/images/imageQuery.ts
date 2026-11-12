import type { ItemFilterQuery } from '$lib/types_itemFilter';
import type { ImageViewParams, NearDupFilter } from '$lib/types_images';

/** The gallery request: the workspace's item filter plus the image-only parameters it holds. */
export function imageListQuery(
  item: ItemFilterQuery,
  view: { unlabeledOnly: boolean; sort: string | null; nearDup: NearDupFilter | null },
): ItemFilterQuery & ImageViewParams {
  return {
    ...item,
    ...(view.unlabeledOnly ? { label_state: 'has_unlabeled' as const } : {}),
    ...(view.nearDup ? { near_dup: view.nearDup } : {}),
    ...(view.sort ? { sort: view.sort } : {}),
  };
}
