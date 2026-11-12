/**
 * Source images: the served image page (`GET /images`), one image with its items and boxes
 * (`GET /images/{id}`), and the image-only list parameters. Pinned to the vendored OpenAPI by
 * `contract/imagesContract.test.ts`.
 */
import type { BoxTone, Crop } from '$lib/types';

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
