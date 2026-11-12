/**
 * Source images: the gallery (`GET {prefix}/images`, a cursored page of whole images with their
 * served counts and boxes), one image with its items and served boxes, and the URL of its pixels.
 * Types in `$lib/types_images`.
 */
import { apiBase, apiFetch, qs, scoped } from '$lib/api';
import { mapRawCrop, type RawCrop } from '$lib/api_cropMap';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import type { ImageDetailResponse, ImagePage, ImageViewParams } from '$lib/types_images';

/** `GET /images`: the shared item filter plus the image-only parameters, one page at `cursor`. */
export function listImages(
  query: ItemFilterQuery & ImageViewParams,
  cursor: string | null,
  signal?: AbortSignal,
): Promise<ImagePage> {
  return apiFetch<ImagePage>(`${scoped()}/images${qs({ ...query, cursor })}`, {}, signal);
}

/**
 * `GET {API_PREFIX}/images/{id}` — one source image: its facts, its counts,
 * every item detected on it (siblings included) and the boxes to draw. The
 * pixels are `getImageFileUrl`; `SourceImageOverlay` draws over them.
 */
export async function getImage(
  imageId: string,
  signal?: AbortSignal,
): Promise<ImageDetailResponse> {
  type Raw = Omit<ImageDetailResponse, 'items'> & { items: RawCrop[] };
  const raw = await apiFetch<Raw>(
    `${scoped()}/images/${encodeURIComponent(imageId)}`,
    {},
    signal,
  );
  return { ...raw, items: raw.items.map(mapRawCrop) };
}

/**
 * The clean source pixels of an image, shrunk to ~`maxDim`px on the longest
 * side so a review view does not pull 2+ MB per open. Nothing is drawn on
 * them: `SourceImageOverlay` draws the served boxes.
 */
export function getImageFileUrl(imageId: string, maxDim: number = 1280): string {
  return `${apiBase}${scoped()}/images/${encodeURIComponent(imageId)}/file?max_dim=${maxDim}`;
}
