/** Served `GET /images` bodies (`ImagePage`, `ImageCardWire`) for tests: every key present. */
import { API_PREFIX } from '$lib/api';
import type { ImageCardWire, ImagePage } from '$lib/types_images';

export function imageCard(over: Partial<ImageCardWire> = {}): ImageCardWire {
  return {
    image_id: 'img-1',
    file_name: 'img-1.jpg',
    thumbnail_url: `${API_PREFIX}/projects/p/images/img-1/thumbnail`,
    width: 640,
    height: 480,
    source: 'test',
    indexed_at: null,
    taken_at: null,
    has_location: false,
    counts: { items: 0, labeled: 0, validated: 0, unlabeled: 0, in_holdout: 0 },
    near_dup: null,
    boxes: [],
    boxes_truncated: false,
    ...over,
  };
}

export function imagePage(
  items: ImageCardWire[] = [],
  over: Partial<ImagePage> = {},
): ImagePage {
  return {
    items,
    next_cursor: null,
    total: items.length,
    limit: 60,
    join_truncated: false,
    sort_applied: 'indexed_at:desc',
    sort_options: [{ id: 'indexed_at:desc', label: 'Newest first' }],
    hidden: { excluded: 0, test_holdout: 0 },
    empty_reason: null,
    empty_message: null,
    ...over,
  };
}
