/**
 * Where the Review view reads its items. A served queue (`ws.url.queue`) is read from
 * `GET /review/{queue}` in that queue's own order; with no queue the view walks the Workspace
 * query on `GET /crops`, in the Detections view's served sort (D-18). Either way the server
 * pages and locates, so one request finds the focused item at any depth.
 */
import { getCrops, locateCrop } from '$lib/api_cropRead';
import { getReviewQueue, locateInReviewQueue } from '$lib/api_review';
import type { CropFilter } from '$lib/types';
import type { CursorPage } from '$lib/types_cursor';
import type { ReviewItem } from '$lib/types_review';

export interface ReviewSourceQuery {
  /** A served queue id, or null to walk the query on `/crops`. */
  queue: string | null;
  /** The shared filter spelled as query parameters (`ItemFilterState.toQuery`). */
  filter: CropFilter;
  /** `sort_review` for a queue, `sort_detections` for `/crops`; null takes the server's default. */
  sort: string | null;
}

/** Where one item sits in the source's order, the same shape from either route. */
export interface ReviewLocation {
  found: boolean;
  index: number | null;
  total: number | null;
  /** Starts a page at the item; null for the first row. */
  cursorBefore: string | null;
  /** The served reason it is not in the list, with its sentence when the route serves one. */
  reason: string | null;
  reasonMessage: string | null;
}

export interface ReviewSource {
  readonly kind: 'queue' | 'crops';
  page(
    cursor: string | null,
    limit: number,
    signal?: AbortSignal,
  ): Promise<CursorPage<ReviewItem>>;
  locate(cropId: string, signal?: AbortSignal): Promise<ReviewLocation>;
}

const withSort = (filter: CropFilter, sort: string | null): CropFilter =>
  sort ? { ...filter, sort } : filter;

function queueSource(queue: string, { filter, sort }: ReviewSourceQuery): ReviewSource {
  const params = withSort(filter, sort) as Record<string, unknown>;
  return {
    kind: 'queue',
    page: (cursor, limit, signal) => getReviewQueue(queue, cursor, limit, params, signal),
    async locate(cropId, signal) {
      const at = await locateInReviewQueue(queue, cropId, params, signal);
      return {
        found: at.in_queue,
        index: at.index,
        total: at.total,
        cursorBefore: at.cursor_before,
        reason: at.reason,
        reasonMessage: null,
      };
    },
  };
}

function cropsSource({ filter, sort }: ReviewSourceQuery): ReviewSource {
  const params = withSort(filter, sort);
  return {
    kind: 'crops',
    async page(cursor, limit, signal) {
      const page = await getCrops({ ...params, limit }, cursor, signal);
      return { ...page, items: page.items.map((crop) => ({ ...crop, reason: null })) };
    },
    async locate(cropId, signal) {
      const at = await locateCrop(cropId, params, signal);
      return {
        found: at.in_view,
        index: at.index,
        total: at.total,
        cursorBefore: at.cursor_before,
        reason: at.reason,
        reasonMessage: at.reason_message,
      };
    },
  };
}

export function reviewSource(query: ReviewSourceQuery): ReviewSource {
  return query.queue ? queueSource(query.queue, query) : cropsSource(query);
}
