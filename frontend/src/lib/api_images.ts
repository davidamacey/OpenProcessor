/**
 * The image gallery (`GET {prefix}/images`): a cursored page of whole images with their served
 * counts and boxes. Types in `$lib/types_images`.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type { ItemFilterQuery } from '$lib/types_itemFilter';
import type { ImagePage, ImageViewParams } from '$lib/types_images';

/** `GET /images`: the shared item filter plus the image-only parameters, one page at `cursor`. */
export function listImages(
  query: ItemFilterQuery & ImageViewParams,
  cursor: string | null,
  signal?: AbortSignal,
): Promise<ImagePage> {
  return apiFetch<ImagePage>(`${scoped()}/images${qs({ ...query, cursor })}`, {}, signal);
}
