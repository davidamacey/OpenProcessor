/** A served `GET /images/{id}` body (`ImageDetailResponse`) for component tests: every key
 *  present, `image` and `items` taken from the test. */
import type { Crop, ImageDetailResponse } from '$lib/types';

export function imageDetail(
  over: { image?: Record<string, unknown>; items?: Crop[] } = {},
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
    boxes: [],
    items_truncated: false,
  };
}
