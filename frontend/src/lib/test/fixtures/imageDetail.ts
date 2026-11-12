/** A served `GET /images/{id}` body (`ImageDetailResponse`) for component tests: every key
 *  present, `image`, `items` and `boxes` taken from the test. */
import type { Crop } from '$lib/types';
import type { ImageBox, ImageDetailResponse } from '$lib/types_images';

export function imageDetail(
  over: { image?: Record<string, unknown>; items?: Crop[]; boxes?: ImageBox[] } = {},
): ImageDetailResponse {
  return {
    image: {
      image_id: 'img-1',
      file_name: 'img-1.jpg',
      file_url: '/images/img-1/file',
      width: 100,
      height: 100,
      source: 'test',
      indexed_at: null,
      taken_at: null,
      location: null,
      dataset_split: null,
      in_holdout: false,
      near_dup: null,
      ...over.image,
    },
    counts: { items: 0, labeled: 0, validated: 0, unlabeled: 0, in_holdout: 0 },
    items: over.items ?? [],
    boxes: over.boxes ?? [],
    items_truncated: false,
  };
}

/** A served `BoxView` with every key present. */
export function imageBox(over: Partial<ImageBox> = {}): ImageBox {
  return {
    crop_id: 'crop-1',
    box_id: null,
    kind: 'item',
    bbox_norm: [0.2, 0.2, 0.4, 0.4],
    label: 'widget_a',
    label_kind: 'class',
    class_name: 'widget_a',
    tone: 'validated',
    state: null,
    locked: false,
    lock_label: null,
    geometry_revision: null,
    ...over,
  };
}
