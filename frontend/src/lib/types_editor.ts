/**
 * Wire shapes of the image-editor routes: draw a box (`POST /images/{id}/items`), move or
 * resize it (`PATCH /crops/{id}/geometry`) and its undo, and the locate answer the filmstrip
 * walks. Pinned to the vendored OpenAPI by `contract/editorContract.test.ts`.
 */
import type { ImageBox, ImageCounts } from '$lib/types_images';
import type { XYXY } from '$lib/annotations/types';

/** `ItemCreateRequest`: a drawn box becomes an item with a person's class and geometry. */
export interface ItemCreateBody {
  bbox_norm: XYXY;
  class_name: string;
  /** Sent again unchanged, it returns the first item instead of a second. */
  request_id: string;
}

/** `ItemCreated`, with the item mapped to a `Crop` by the client. */
export interface ItemCreatedWire<Item> {
  item: Item;
  created: boolean;
  image_boxes: ImageBox[];
  image_counts: ImageCounts;
  embedding_state: string;
  embedding_message: string | null;
}

/** `GeometryPatchRequest`. */
export interface GeometryPatchBody {
  bbox_norm: XYXY;
  expected_geometry_revision: number;
  /** Region boxes that fall outside the new box and may be deleted with it. */
  drop_region_box_ids?: string[];
}

/** `GeometryWritten`, with the item mapped to a `Crop` by the client. */
export interface GeometryWrittenWire<Item> {
  item: Item;
  image_boxes: ImageBox[];
  embedding_state: string;
  embedding_message: string | null;
}

/** `ImageLocate`: where one image sits in the gallery order, and the cursors around it. */
export interface ImageLocate {
  image_id: string;
  in_view: boolean;
  index: number | null;
  total: number;
  /** The cursor of the page that starts at this image. */
  cursor_before: string | null;
  /** The cursor of the page after it. */
  cursor_after: string | null;
  /** The image right before this one, for walking back; null for the first row. */
  previous_image_id: string | null;
  reason: 'not_found' | 'filtered_out' | null;
  reason_message: string | null;
}
