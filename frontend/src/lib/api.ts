/**
 * Typed API client for the openprocessor `/curation/` endpoints.
 *
 * - Single base URL, defaulting to `http://localhost:4603`.
 * - `apiFetch` retries on 5xx with exponential backoff (3 tries, 250 / 500 / 1000ms).
 * - Caller-supplied AbortSignal is honoured; cancellation never retries.
 * - On 4xx the original ApiError is thrown immediately (no retry).
 *
 * All endpoint URL patterns come from Section "Phase 2D" of the v7 plan.
 */

import type {
  BulkLabelResult,
  ClusterFilter,
  CropFilter,
  OpClass,
  OpCluster,
  OpCrop,
  OpHealth,
  OpStats,
  PaginatedResponse,
  ReviewItem,
  ReviewTab,
} from './types';

// Vite exposes only PUBLIC_-prefixed env vars to the client. SvelteKit uses
// `$env/dynamic/public` but importing that here would force every consumer
// onto SSR-only paths; we go through `import.meta.env` so this module remains
// usable in pure client contexts (and, for production, the value is baked in
// at build time and overridable via the docker-entrypoint shim).
const RAW_BASE =
  (import.meta.env?.PUBLIC_TRITON_API_URL as string | undefined) ?? 'http://localhost:4603';

export const apiBase: string = RAW_BASE.replace(/\/+$/, '');

export class ApiError extends Error {
  status: number;
  body: unknown;
  url: string;
  constructor(status: number, url: string, body: unknown, message?: string) {
    super(message ?? `API ${status} ${url}`);
    this.status = status;
    this.body = body;
    this.url = url;
  }
}

const RETRY_DELAYS_MS = [250, 500, 1000];

async function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const id = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(id);
        reject(new DOMException('Aborted', 'AbortError'));
      },
      { once: true },
    );
  });
}

export async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const url = path.startsWith('http') ? path : `${apiBase}${path}`;
  let attempt = 0;
  let lastError: unknown;
  // 1 initial + 3 retries on 5xx => 4 attempts max.
  for (; attempt < RETRY_DELAYS_MS.length + 1; attempt++) {
    try {
      const res = await fetch(url, {
        ...init,
        signal: signal ?? init.signal ?? null,
        headers: {
          Accept: 'application/json',
          ...(init.body ? { 'Content-Type': 'application/json' } : {}),
          ...(init.headers ?? {}),
        },
      });
      if (res.ok) {
        if (res.status === 204) return undefined as T;
        const ct = res.headers.get('content-type') ?? '';
        if (ct.includes('application/json')) return (await res.json()) as T;
        return (await res.text()) as unknown as T;
      }
      let body: unknown = null;
      try {
        body = await res.json();
      } catch {
        try {
          body = await res.text();
        } catch {
          /* ignore */
        }
      }
      const err = new ApiError(res.status, url, body);
      // Don't retry on 4xx — they won't get better.
      if (res.status < 500) throw err;
      lastError = err;
    } catch (e) {
      if (e instanceof ApiError && e.status < 500) throw e;
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      lastError = e;
    }
    if (attempt < RETRY_DELAYS_MS.length) {
      try {
        await sleep(RETRY_DELAYS_MS[attempt]!, signal);
      } catch (e) {
        throw e;
      }
    }
  }
  throw lastError ?? new Error(`apiFetch failed: ${url}`);
}

// -- query string helpers ------------------------------------------------

function qs(params: Record<string, unknown>): string {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null) continue;
    u.set(k, String(v));
  }
  const s = u.toString();
  return s ? `?${s}` : '';
}

// -- endpoints -----------------------------------------------------------

export function getHealth(signal?: AbortSignal): Promise<OpHealth> {
  return apiFetch<OpHealth>('/curation/health', {}, signal);
}

export function getStats(signal?: AbortSignal): Promise<OpStats> {
  return apiFetch<OpStats>('/curation/stats/dataset', {}, signal);
}

export function getClasses(signal?: AbortSignal): Promise<OpClass[]> {
  return apiFetch<OpClass[]>('/curation/classes', {}, signal);
}

export function getClusters(
  filter: ClusterFilter = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<OpCluster>> {
  // Plan Phase 2D: clusters are exposed via /clusters/stats/op_vehicles.
  // The stats endpoint returns the same paginated cluster list our UI needs.
  return apiFetch<PaginatedResponse<OpCluster>>(
    `/clusters/stats/op_vehicles${qs({ ...filter })}`,
    {},
    signal,
  );
}

export function getCluster(
  id: number,
  page = 1,
  pageSize = 60,
  signal?: AbortSignal,
): Promise<{ cluster: OpCluster; crops: PaginatedResponse<OpCrop> }> {
  return apiFetch<{ cluster: OpCluster; crops: PaginatedResponse<OpCrop> }>(
    `/clusters/op_vehicles/${id}${qs({ page, page_size: pageSize })}`,
    {},
    signal,
  );
}

export function getCrops(
  filter: CropFilter = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<OpCrop>> {
  return apiFetch<PaginatedResponse<OpCrop>>(`/curation/crops${qs({ ...filter })}`, {}, signal);
}

export function putCropLabel(
  cropId: string,
  classId: number,
  signal?: AbortSignal,
): Promise<OpCrop> {
  return apiFetch<OpCrop>(
    `/curation/crops/${encodeURIComponent(cropId)}/label`,
    {
      method: 'PUT',
      body: JSON.stringify({ class_id: classId, validated: true }),
    },
    signal,
  );
}

export function bulkLabel(
  cropIds: string[],
  classId: number,
  signal?: AbortSignal,
): Promise<BulkLabelResult> {
  return apiFetch<BulkLabelResult>(
    '/curation/crops/batch_label',
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds, class_id: classId, validated: true }),
    },
    signal,
  );
}

/** Undo: reset crop label back to its model-suggested value. */
export function deleteCropLabel(cropId: string, signal?: AbortSignal): Promise<void> {
  return apiFetch<void>(
    `/curation/crops/${encodeURIComponent(cropId)}/label`,
    { method: 'DELETE' },
    signal,
  );
}

export function runGemmaOnCluster(
  clusterId: number,
  signal?: AbortSignal,
): Promise<{ enqueued: number }> {
  return apiFetch<{ enqueued: number }>(
    '/curation/gemma/label_batch',
    {
      method: 'POST',
      body: JSON.stringify({ cluster_id: clusterId, only_unvalidated: true }),
    },
    signal,
  );
}

export function refineCluster(
  clusterId: number,
  signal?: AbortSignal,
): Promise<{ subclusters: number }> {
  return apiFetch<{ subclusters: number }>(
    `/curation/clusters/refine/${clusterId}`,
    { method: 'POST' },
    signal,
  );
}

export function getReviewQueue(
  tab: ReviewTab,
  page = 1,
  pageSize = 30,
  filter: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<ReviewItem>> {
  return apiFetch<PaginatedResponse<ReviewItem>>(
    `/curation/review/${tab}${qs({ page, page_size: pageSize, ...filter })}`,
    {},
    signal,
  );
}

export function exportYolo(signal?: AbortSignal): Promise<{ job_id: string }> {
  return apiFetch<{ job_id: string }>('/curation/export/yolo', { method: 'POST' }, signal);
}

// -- image URL helpers (no fetch — used directly in <img src=...>) -------

export function getThumbUrl(cropId: string): string {
  return `${apiBase}/curation/crops/${encodeURIComponent(cropId)}/thumbnail`;
}

export function getSourceImageUrl(cropId: string): string {
  return `${apiBase}/curation/crops/${encodeURIComponent(cropId)}/image`;
}

export function getSourceImageWithBbox(cropId: string): string {
  return `${apiBase}/curation/crops/${encodeURIComponent(cropId)}/image?bbox=1`;
}
