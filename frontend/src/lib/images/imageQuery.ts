import type { ItemFilter, ItemFilterQuery } from '$lib/types_itemFilter';
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

/** The bulk-action filter for selected images: the current filter narrowed to those images. */
export function imageSelectionFilter(body: ItemFilter, imageIds: string[]): ItemFilter {
  return { ...body, image_ids: imageIds };
}
