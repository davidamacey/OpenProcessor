/**
 * Item reads: crop pages, one crop, its history and source-image context,
 * text search, and the thumbnail / source-image URLs.
 */
import { apiBase, apiFetch, qs, scoped } from '$lib/api';
import { mapRawCrop, type RawCrop } from '$lib/api_cropMap';
import type {
  Crop,
  CropContextResponse,
  CropFilter,
  CropHistoryResponse,
  PaginatedResponse,
} from '$lib/types';
import type { SearchCrop } from '$lib/types_review';
import type { ReprocessRequest } from '$lib/types_import';

export async function getCrops(
  filter: CropFilter = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<Crop>> {
  type Raw = {
    total: number;
    page: number;
    page_size: number;
    crops: RawCrop[];
    n_unembedded?: number | null;
    suggested_reprocess?: ReprocessRequest | null;
  };
  const raw = await apiFetch<Raw>(`${scoped()}/crops${qs({ ...filter })}`, {}, signal);
  return {
    items: raw.crops.map(mapRawCrop),
    total: raw.total,
    page: raw.page,
    page_size: raw.page_size,
    n_unembedded: raw.n_unembedded ?? null,
    suggested_reprocess: raw.suggested_reprocess ?? null,
  };
}

/**
 * `GET {API_PREFIX}/crops/{id}/history` — the item's label-write history,
 * oldest first. Untyped on the wire beyond `crop_id`/`entries`
 * (`additionalProperties: true`); each entry's other keys are read
 * tolerantly by the caller (`CropHistoryEntry`'s fields are all
 * optional). Used by `CropMetaPanel`'s lazy-on-open history section (G7).
 */
export function getCropHistory(
  cropId: string,
  signal?: AbortSignal,
): Promise<CropHistoryResponse> {
  return apiFetch<CropHistoryResponse>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/history`,
    {},
    signal,
  );
}

/**
 * `GET {API_PREFIX}/crops/{id}/context` — the crop's shared source image
 * metadata plus every item cropped from it (siblings, including the
 * requested crop). The image itself is served by `/crops/{id}/image`
 * (`getSourceImageScaled`) — as of K6 this is a
 * clean image with no server-drawn boxes; `SourceImageOverlay.svelte`
 * draws every box/label from this response's `items`.
 */
export async function getCropContext(
  cropId: string,
  signal?: AbortSignal,
): Promise<CropContextResponse> {
  type Raw = {
    image: CropContextResponse['image'];
    items: RawCrop[];
  };
  const raw = await apiFetch<Raw>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/context`,
    {},
    signal,
  );
  return { image: raw.image, items: raw.items.map(mapRawCrop) };
}

/**
 * Fetch a single crop by id from the authoritative store. Used by the
 * review-page "Back" path so the operator sees what was actually
 * persisted rather than a possibly-stale local snapshot. Endpoint:
 * `GET {API_PREFIX}/crops/{crop_id}`.
 */
export async function getCrop(cropId: string, signal?: AbortSignal): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}`,
    {},
    signal,
  );
  return mapRawCrop(raw);
}

/**
 * Free-text semantic search over crops (P2-14). Backend:
 * `GET {API_PREFIX}/search/text`, gated behind the `semantic_search` overlay in
 * `{API_PREFIX}/methods` (see `isSemanticSearchAvailable` in `./strategies`) — a
 * caller must check that gate before rendering a UI that calls this.
 *
 * Mirrors `getReviewQueue`'s response-shape handling: the endpoint
 * returns `{items, total, page, page_size}` with items shaped like other
 * crop payloads plus a similarity score, normalized through `mapRawCrop`
 * so every existing crop-consuming component (CropCard, thumbnails,
 * label actions) keeps working unchanged against a search result.
 *
 * `filter` threads through arbitrary extra query params — same
 * `Record<string, unknown>` convention as `getReviewQueue` — so callers
 * can pass `cluster_id` (search scoped to one cluster on
 * `/clusters/[id]`) or the active review tab/filters (on `/review`)
 * without this function needing to know their shape.
 */
export async function searchCrops(
  q: string,
  page = 1,
  pageSize = 30,
  filter: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<SearchCrop>> {
  const raw = await apiFetch<RawSearchPage>(
    `${scoped()}/search/text${qs({ q, page, page_size: pageSize, ...filter })}`,
    {},
    signal,
  );
  return mapSearchPage(raw, page, pageSize);
}

/**
 * "More like this": crops nearest to the stored vector of `cropId`. Backend:
 * `GET {API_PREFIX}/search/similar` (same gate and page shape as `searchCrops`).
 * A crop that is unknown or has no vector is refused 404/409 with a served message.
 */
export async function searchSimilarCrops(
  cropId: string,
  page = 1,
  pageSize = 30,
  filter: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<SearchCrop>> {
  const raw = await apiFetch<RawSearchPage>(
    `${scoped()}/search/similar${qs({ crop_id: cropId, page, page_size: pageSize, ...filter })}`,
    {},
    signal,
  );
  return mapSearchPage(raw, page, pageSize);
}

/** A normalized `[x1, y1, x2, y2]` region of a picture. */
export type ImageSearchBox = [number, number, number, number];

/** The picture a query-by-image search ranks by: an upload, or an image already in the project. */
export interface ImageSearchQuery {
  file?: File;
  imageId?: string;
  box?: ImageSearchBox | null;
}

/**
 * Search by a picture. Backend: `POST {API_PREFIX}/search/image` (multipart `image` or `image_id`,
 * optional `box`); same gate, filters and page shape as `searchCrops`.
 */
export async function searchImageCrops(
  query: ImageSearchQuery,
  page = 1,
  pageSize = 30,
  filter: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<SearchCrop>> {
  const body = new FormData();
  if (query.file) body.append('image', query.file, query.file.name);
  if (query.imageId) body.append('image_id', query.imageId);
  if (query.box) body.append('box', query.box.join(','));
  const raw = await apiFetch<RawSearchPage>(
    `${scoped()}/search/image${qs({ page, page_size: pageSize, ...filter })}`,
    { method: 'POST', body },
    signal,
  );
  return mapSearchPage(raw, page, pageSize);
}

type RawSearchItem = RawCrop & {
  semantic_score: number | null;
  matched_box_id?: string | null;
  matched_box_stale?: boolean | null;
};
type RawSearchPage = {
  total: number;
  page: number;
  page_size: number;
  items: RawSearchItem[];
  unembedded_in_scope?: number | null;
};

function mapSearchPage(
  raw: RawSearchPage,
  page: number,
  pageSize: number,
): PaginatedResponse<SearchCrop> {
  // A crop can be served once per matching box; the result grids key by
  // crop id, so keep the first (best-ranked) hit per id.
  const seen = new Set<string>();
  const hits = (raw.items ?? []).filter((it) => {
    if (seen.has(it.crop_id)) return false;
    seen.add(it.crop_id);
    return true;
  });
  const items: SearchCrop[] = hits.map((it) => {
    const base = mapRawCrop(it);
    return {
      ...base,
      // The backend's `_hydrate_item` (OpenProcessor semantic_search.py) sends
      // the match score as `semantic_score`.
      similarity_score: it.semantic_score ?? 0,
      // `target=box` only: the region of the item whose vector matched.
      matched_box_id: it.matched_box_id ?? null,
      matched_box_stale: it.matched_box_stale ?? null,
    };
  });
  return {
    items,
    total: raw.total ?? items.length,
    page: raw.page ?? page,
    page_size: raw.page_size ?? pageSize,
    unembedded_in_scope: raw.unembedded_in_scope ?? null,
  };
}

// -- image URL helpers (no fetch — used directly in <img src=...>) -------

/**
 * URL for a crop thumbnail. Defaults to 160px — small enough to load
 * fast for fast grid scanning of thousands of crops, large enough that
 * subject details (color, shape, fine texture) remain readable.
 * Server-side aspect-correct rendering preserves the bbox proportions.
 *
 * Pass a larger ``size`` (256-512) for click-to-inspect / focused review
 * where rendering quality matters more than transfer speed.
 */
export function getThumbUrl(cropId: string, size: number = 160): string {
  return `${apiBase}${scoped()}/crops/${encodeURIComponent(cropId)}/thumbnail?size=${size}`;
}

/**
 * Source image downscaled to ~`maxDim`px on the longest side — plenty
 * for `SourceImageOverlay`'s client-drawn boxes to be readable without
 * pushing 2+ MB per open. K6 (docs/design/
 * k6-frontend-overlay-plan-2026-09-24.md): the backend no longer burns
 * any box/label into this image (`getSourceImageWithBbox` — the
 * server-overlay-era name — is retired; `SourceImageOverlay` draws
 * everything itself from `getCropContext`).
 */
export function getSourceImageScaled(cropId: string, maxDim: number = 1280): string {
  return `${apiBase}${scoped()}/crops/${encodeURIComponent(cropId)}/image?max_dim=${maxDim}`;
}
