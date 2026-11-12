import type { ItemFilterQuery } from '$lib/types_itemFilter';
import type { ImageViewParams } from '$lib/types_images';

/** The gallery request: the workspace's item filter plus the image-only parameters it holds. */
export function imageListQuery(
  item: ItemFilterQuery,
  view: { unlabeledOnly: boolean; sort: string | null },
): ItemFilterQuery & ImageViewParams {
  return {
    ...item,
    ...(view.unlabeledOnly ? { label_state: 'has_unlabeled' as const } : {}),
    ...(view.sort ? { sort: view.sort } : {}),
  };
}
