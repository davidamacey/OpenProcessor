/**
 * Source images: the served image page (`GET /images`), one image with its items and boxes
 * (`GET /images/{id}`), and the image-only list parameters. Pinned to the vendored OpenAPI by
 * `contract/imagesContract.test.ts`.
 */
import type { BoxTone, Crop, HiddenCounts } from '$lib/types';
import type { SortOption } from '$lib/types_cursor';

export interface ImageBox {
  crop_id: string;
  box_id: string | null;
  kind: 'item' | 'region';
  bbox_norm: [number, number, number, number];
  label: string;
  label_kind: 'class' | 'proposed' | 'detector' | 'region' | 'none';
  class_name: string | null;
  tone: BoxTone;
  state: string | null;
  locked: boolean;
  lock_label: string | null;
  geometry_revision: number | null;
}

export interface ImageCounts {
  items: number;
  labeled: number;
  validated: number;
  unlabeled: number;
  in_holdout: number;
}

export interface ImageNearDup {
  group_id: string;
  purpose: 'dedup' | 'leak';
  size: number | null;
  is_representative: boolean | null;
}

export interface ImageInfo {
  image_id: string;
  file_name: string;
  file_url: string;
  width: number | null;
  height: number | null;
  source: string;
  indexed_at: string | null;
  taken_at: string | null;
  location: { latitude: number; longitude: number; precision: string | null } | null;
  dataset_split: string | null;
  in_holdout: boolean;
  near_dup: ImageNearDup | null;
}

/** `GET {API_PREFIX}/images/{id}` — one source image, its counts, every item
 *  detected on it and the boxes to draw. */
export interface ImageDetailResponse {
  image: ImageInfo;
  counts: ImageCounts;
  items: Crop[];
  boxes: ImageBox[];
  items_truncated: boolean;
}
export type LabelState = 'has_unlabeled' | 'all_labeled' | 'no_items';
export type NearDupFilter = 'member' | 'representative' | 'none';

/** The image-only list parameters (`ImageViewQuery`), on top of the shared item filter. */
export interface ImageViewParams {
  source?: string | null;
  in_holdout?: boolean | null;
  near_dup?: NearDupFilter | null;
  label_state?: LabelState | null;
  /** One of the served `sort_options` ids; omitted, the server's default applies. */
  sort?: string | null;
  include_test?: boolean;
  /** Page size, 1 to 200; omitted, the server's default applies. */
  limit?: number;
}

/** One card of `GET /images`: the facts, the served counts and the boxes to draw. */
export interface ImageCardWire {
  image_id: string;
  file_name: string;
  thumbnail_url: string;
  width: number | null;
  height: number | null;
  source: string;
  indexed_at: string | null;
  taken_at: string | null;
  has_location: boolean;
  counts: ImageCounts;
  near_dup: ImageNearDup | null;
  boxes: ImageBox[];
  boxes_truncated: boolean;
}

export type ImageEmptyReason = 'no_images' | 'no_match';

/** `GET {prefix}/images`: one cursored page of the project's images. */
export interface ImagePage {
  items: ImageCardWire[];
  next_cursor: string | null;
  total: number;
  limit: number;
  join_truncated: boolean;
  sort_applied: string;
  sort_options: SortOption[];
  hidden: HiddenCounts;
  empty_reason: ImageEmptyReason | null;
  empty_message: string | null;
}
