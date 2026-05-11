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
  OpClassCreate,
  OpClassMerge,
  OpClassUpdate,
  OpCluster,
  OpCrop,
  OpExportResult,
  OpExportStatus,
  OpHealth,
  OpModelsStatus,
  OpStats,
  OpTestHoldoutFreezeResult,
  OpTestHoldoutStats,
  PaginatedResponse,
  ReviewItem,
  ReviewTab,
} from './types';
import type {
  CancelResponse,
  LogTailResponse,
  PresetsResponse,
  PreflightReport,
  ProfilesResponse,
  PromoteRequest,
  PromoteResponse,
  RunsListResponse,
  StartCampaignResponse,
  StartTrainResponse,
  TrainCampaignSpec,
  TrainJobSpec,
  TrainJobStatus,
} from './types_train';

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

// -- plates browse / training-cohort selection ---------------------------

export interface PlateBrowseItem {
  crop_id: string;
  id: string;
  image_path: string;
  bbox_norm: number[];
  plate_bbox_norm: number[] | null;
  plate_score: number | null;
  plate_status: string | null;
  plate_verified: boolean | null;
  plate_detector: string | null;
  plate_detector_version: string | null;
  plate_detector_chain: string[] | null;
  plate_bbox_frame: string | null;
  plate_detected_at: string | null;
  plate_verifier: string | null;
  plate_verifier_version: string | null;
  plate_verified_at: string | null;
  plate_rejection_reason: string | null;
  plate_visible: boolean | null;
  plate_text: string | null;
  plate_text_source: string | null;
  plate_text_confidence: number | null;
  class_id: number | null;
  class_name: string | null;
  cluster_id: number | null;
  updated_at: string;
  thumbnail_url?: string;
  plate_thumbnail_url?: string;
  selection_reason?: string;
}

export interface PlatesPage {
  total: number;
  page: number;
  page_size: number;
  items: PlateBrowseItem[];
  mode?: string;
  selection_reason?: string;
}

export interface PlatesQuery {
  page?: number;
  page_size?: number;
  class_id?: number;
  cluster_id?: number;
  min_score?: number;
  max_score?: number;
  verified?: boolean;
  detector?: string;
  text?: string;
  include_test?: boolean;
}

export function getPlates(params: PlatesQuery = {}, signal?: AbortSignal): Promise<PlatesPage> {
  return apiFetch<PlatesPage>(`/curation/plates${qs(params as Record<string, unknown>)}`, {}, signal);
}

export type TrainingCohortMode =
  | 'lpr_blind_spots'
  | 'lpr_low_conf_correct'
  | 'disagreement'
  | 'human_corrected';

export function getTrainingCandidates(
  mode: TrainingCohortMode,
  params: { page?: number; page_size?: number; class_id?: number } = {},
  signal?: AbortSignal,
): Promise<PlatesPage> {
  return apiFetch<PlatesPage>(
    `/curation/plates/training_candidates${qs({ mode, ...params })}`,
    {},
    signal,
  );
}

export function getModelsStatus(signal?: AbortSignal): Promise<OpModelsStatus> {
  return apiFetch<OpModelsStatus>('/curation/models/status', {}, signal);
}

export async function getStats(signal?: AbortSignal): Promise<OpStats> {
  // The API returns
  //   /curation/stats/dataset:  {total_crops, validated, test_holdout, by_source}
  //   /curation/stats/classes:  {classes:[{class_id, class_name, count, validated_count}, ...]}
  // The labeler dashboard expects OpStats which uses validated_crops /
  // test_holdout_crops / per_class / ingestion.* — fold the two server
  // payloads into that shape so the dashboard can render directly.
  type RawDataset = {
    total_crops?: number;
    validated?: number;
    test_holdout?: number;
    by_source?: Array<{ key: string; doc_count: number }>;
  };
  type RawClasses = {
    classes?: Array<{
      class_id: number;
      class_name: string;
      count?: number;
      sample_count?: number;
      validated_count?: number;
    }>;
  };
  const [ds, cls] = await Promise.all([
    apiFetch<RawDataset>('/curation/stats/dataset', {}, signal),
    apiFetch<RawClasses>('/curation/stats/classes', {}, signal).catch(() => ({ classes: [] })),
  ]);
  const totalImages = (ds.by_source ?? []).reduce((acc, b) => acc + (b.doc_count || 0), 0);
  return {
    total_crops: ds.total_crops ?? 0,
    validated_crops: ds.validated ?? 0,
    test_holdout_crops: ds.test_holdout ?? 0,
    ingestion: {
      images_processed: totalImages,
      images_pending: 0,
      last_run_at: null,
    },
    per_class: (cls.classes ?? []).map((c) => ({
      class_id: c.class_id,
      class_name: c.class_name,
      count: c.count ?? c.sample_count ?? 0,
      validated_count: c.validated_count ?? 0,
    })),
  };
}

export async function getClasses(signal?: AbortSignal): Promise<OpClass[]> {
  // The API returns `{classes: [{class_id, class_name, group, sample_count,
  // validated_count, deprecated}, ...]}`. Map to the labeler's OpClass
  // shape, which uses `id`/`name`/`count`.
  type RawClass = {
    class_id?: number;
    id?: number;
    class_name?: string;
    name?: string;
    group?: string | null;
    sample_count?: number;
    count?: number;
    validated_count?: number;
    color?: string | null;
    deprecated?: boolean;
    added_at?: string;
    hotkey_letter?: string | null;
  };
  const res = await apiFetch<{ classes: RawClass[] } | RawClass[]>('/curation/classes', {}, signal);
  const raw = Array.isArray(res) ? res : res.classes ?? [];
  return raw.map((c) => ({
    id: c.class_id ?? c.id ?? -1,
    name: c.class_name ?? c.name ?? '',
    group: c.group ?? null,
    count: c.sample_count ?? c.count ?? 0,
    validated_count: c.validated_count ?? 0,
    added_at: c.added_at ?? '',
    color: c.color ?? null,
    deprecated: !!c.deprecated,
    hotkey_letter: c.hotkey_letter ?? null,
  }));
}

export async function getClusters(
  filter: ClusterFilter = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<OpCluster>> {
  // The existing /clusters/stats/{index} response shape is
  // `{faiss: {...}, opensearch_clusters: [{cluster_id, count}, ...]}`.
  // Map to the labeler's PaginatedResponse<OpCluster>.
  type RawStats = {
    status?: string;
    faiss?: { n_clusters?: number };
    opensearch_clusters?: Array<{ cluster_id: number; count: number }>;
    total_clusters_in_opensearch?: number;
  };
  const raw = await apiFetch<RawStats>(
    `/clusters/stats/op_vehicles${qs({ ...filter })}`,
    {},
    signal,
  );
  const baseItems = (raw.opensearch_clusters ?? []).map((c) => ({
    id: c.cluster_id,
    size: c.count,
    purity: null as number | null,
    dominant_class_id: null as number | null,
    dominant_class_name: null as string | null,
    dominant_pct: null as number | null,
    sub_clusters: 0,
    has_subclusters: false,
    representative_crop_ids: [] as string[],
    updated_at: null as string | null,
  }));
  // Single-call fetch of top-K representatives across all clusters. When
  // a class filter is active, the server narrows the agg to clusters that
  // contain at least one crop of that class.
  const repsQs = qs({
    per_cluster: 4,
    class_id: filter.class_id ?? undefined,
  });
  let reps: Record<string, Array<{ crop_id: string; class_name?: string | null }>> = {};
  try {
    const repsResp = await apiFetch<{
      clusters: Record<string, Array<{ crop_id: string; class_name?: string | null }>>;
    }>(`/curation/clusters/representatives${repsQs}`, {}, signal);
    reps = repsResp.clusters ?? {};
  } catch {
    /* representatives are best-effort; cards still show without thumbs */
  }
  // When filtering by class, drop any cluster the server didn't return reps for.
  const visibleIds =
    filter.class_id != null ? new Set(Object.keys(reps).map((k) => Number(k))) : null;
  const filteredBase = visibleIds
    ? baseItems.filter((c) => visibleIds.has(c.id))
    : baseItems;
  const items: OpCluster[] = filteredBase.map((c) => {
    const r = reps[String(c.id)] ?? [];
    const counts = new Map<string, number>();
    for (const x of r) {
      if (x.class_name) counts.set(x.class_name, (counts.get(x.class_name) ?? 0) + 1);
    }
    let domName: string | null = null;
    let domCount = 0;
    for (const [n, ct] of counts) {
      if (ct > domCount) {
        domName = n;
        domCount = ct;
      }
    }
    const purity = r.length > 0 ? domCount / r.length : null;
    return {
      ...c,
      representative_crop_ids: r.map((x) => x.crop_id),
      dominant_class_name: domName,
      dominant_pct: purity,
      // Purity is derived from the representative sample; with class-based
      // clustering (cluster_id == class_id) every cluster should be 100%
      // pure. Setting it here makes the cluster card's pure/mixed/noisy
      // badge meaningful instead of always showing "noisy 0".
      purity,
    };
  });
  return {
    items,
    total: filter.class_id != null ? items.length : raw.total_clusters_in_opensearch ?? items.length,
    page: 1,
    page_size: items.length,
  };
}

/** Convert API's [x1, y1, x2, y2] to the labeler's BBoxNorm {cx, cy, w, h}. */
function xyxyToBBoxNorm(bb: number[]): import('./types').BBoxNorm {
  const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = bb;
  return {
    cx: (x1 + x2) / 2,
    cy: (y1 + y2) / 2,
    w: Math.max(0, x2 - x1),
    h: Math.max(0, y2 - y1),
  };
}

/** Raw crop shape from the /curation/crops API. */
type RawCrop = {
  crop_id: string;
  image_id?: string;
  image_path: string;
  bbox_norm: number[];
  class_id?: number | null;
  class_name?: string | null;
  class_source?: string;
  confidence?: number;
  cluster_id?: number | null;
  cluster_distance?: number | null;
  label_validated?: boolean;
  label_source?: string;
  plate_bbox_norm?: number[] | null;
  plate_score?: number | null;
  plate_status?: string | null;
  plate_verified?: boolean | null;
  // Provenance fields (Wave 1 — written on every new plate/class write).
  plate_detector?: string | null;
  plate_detector_version?: string | null;
  plate_detector_chain?: string[] | null;
  plate_bbox_frame?: string | null;
  plate_detected_at?: string | null;
  plate_verifier?: string | null;
  plate_verifier_version?: string | null;
  plate_verified_at?: string | null;
  plate_rejection_reason?: string | null;
  plate_visible?: boolean | null;
  plate_text?: string | null;
  plate_text_raw?: string | null;
  plate_text_source?: string | null;
  plate_text_confidence?: number | null;
  plate_text_engine_version?: string | null;
  class_detector?: string | null;
  class_detector_version?: string | null;
  class_labeled_at?: string | null;
  class_labeler?: string | null;
  test_holdout?: boolean;
  thumbnail_url?: string;
  updated_at?: string;
};

// Plate-bbox shape envelope — must match the server-side
// is_plausible_plate_bbox helper in
// openprocessor:src/services/legacy/plate_detect.py. Defense in depth:
// flags rows whose stored bbox is implausible *after* projecting into
// the crop frame, regardless of whether the server-side gate caught it.
function _platePlausibleEnvelope(plate: import('./types').BBoxNorm): boolean {
  const w = plate.w;
  const h = plate.h;
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return false;
  const aspect = w / h;
  if (aspect < 1.2 || aspect > 8.0) return false;
  // In crop-frame coords, w IS plate_w/vehicle_w because the canvas is
  // the vehicle crop. So w > 0.5 → plate covers >50% of vehicle width.
  if (w > 0.5) return false;
  if (w * h > 0.15) return false;
  return true;
}

function _platesShapeWarning(
  plateSrc: number[] | null | undefined,
  vehicleSrc: number[],
): boolean {
  if (!plateSrc || plateSrc.length !== 4) return false;
  const v = xyxyToBBoxNorm(vehicleSrc);
  const vw = v.w;
  const vh = v.h;
  if (vw <= 1e-9 || vh <= 1e-9) return false;
  const [px1 = 0, py1 = 0, px2 = 0, py2 = 0] = plateSrc;
  // Project to crop frame the same way sourceToCropFrame would.
  const cropPlate: import('./types').BBoxNorm = {
    cx: ((px1 + px2) / 2 - (v.cx - vw / 2)) / vw,
    cy: ((py1 + py2) / 2 - (v.cy - vh / 2)) / vh,
    w: (px2 - px1) / vw,
    h: (py2 - py1) / vh,
  };
  return !_platePlausibleEnvelope(cropPlate);
}

function mapRawCrop(c: RawCrop): OpCrop {
  const bb = c.bbox_norm ?? [0, 0, 0, 0];
  const out: OpCrop = {
    id: c.crop_id,
    source_image_path: c.image_path,
    bbox_norm: xyxyToBBoxNorm(bb),
    class_id: c.class_id ?? null,
    class_name: c.class_name ?? null,
    label_source: ((c.label_source || 'model') as OpCrop['label_source']),
    label_validated: !!c.label_validated,
    label_confidence: c.confidence ?? null,
    cluster_id: c.cluster_id ?? null,
    similarity_to_centroid:
      c.cluster_distance != null ? Math.max(0, 1 - c.cluster_distance) : null,
    plate_bbox_norm:
      c.plate_bbox_norm && c.plate_bbox_norm.length === 4
        ? xyxyToBBoxNorm(c.plate_bbox_norm)
        : null,
    plate_score: c.plate_score ?? null,
    plate_status: c.plate_status ?? null,
    plate_verified: c.plate_verified ?? null,
    plate_detector: c.plate_detector ?? null,
    plate_detector_version: c.plate_detector_version ?? null,
    plate_detector_chain: c.plate_detector_chain ?? null,
    plate_bbox_frame: c.plate_bbox_frame ?? null,
    plate_detected_at: c.plate_detected_at ?? null,
    plate_verifier: c.plate_verifier ?? null,
    plate_verifier_version: c.plate_verifier_version ?? null,
    plate_verified_at: c.plate_verified_at ?? null,
    plate_rejection_reason: c.plate_rejection_reason ?? null,
    plate_visible: c.plate_visible ?? null,
    plate_text: c.plate_text ?? null,
    plate_text_raw: c.plate_text_raw ?? null,
    plate_text_source: c.plate_text_source ?? null,
    plate_text_confidence: c.plate_text_confidence ?? null,
    plate_text_engine_version: c.plate_text_engine_version ?? null,
    class_detector: c.class_detector ?? null,
    class_detector_version: c.class_detector_version ?? null,
    class_labeled_at: c.class_labeled_at ?? null,
    class_labeler: c.class_labeler ?? null,
    plate_shape_warning: _platesShapeWarning(c.plate_bbox_norm ?? null, bb),
    test_holdout: !!c.test_holdout,
    // Preserve server-side updated_at — overriding it client-side breaks
    // ordering and lets the same crop key appear twice in keyed each blocks
    // (Svelte each_key_duplicate).
    updated_at:
      typeof (c as Record<string, unknown>).updated_at === 'string'
        ? ((c as Record<string, unknown>).updated_at as string)
        : '',
  };
  return out;
}

export async function getCluster(
  id: number,
  page = 1,
  pageSize = 60,
  signal?: AbortSignal,
): Promise<{ cluster: OpCluster; crops: PaginatedResponse<OpCrop> }> {
  type CropPage = { total: number; page: number; page_size: number; crops: RawCrop[] };
  const cropPage = await apiFetch<CropPage>(
    `/curation/crops${qs({ cluster_id: id, page, page_size: pageSize })}`,
    {},
    signal,
  );
  const items = cropPage.crops.map(mapRawCrop);
  const counts = new Map<string, number>();
  for (const c of items) {
    if (c.class_name) counts.set(c.class_name, (counts.get(c.class_name) ?? 0) + 1);
  }
  let dom_name: string | null = null;
  let dom_count = 0;
  for (const [n, ct] of counts) {
    if (ct > dom_count) {
      dom_name = n;
      dom_count = ct;
    }
  }
  const cluster: OpCluster = {
    id,
    size: cropPage.total,
    purity: cropPage.total > 0 ? dom_count / cropPage.total : null,
    dominant_class_id: null,
    dominant_class_name: dom_name,
    dominant_pct: cropPage.total > 0 ? Math.round((dom_count / cropPage.total) * 100) : null,
    has_subclusters: false,
    representative_crop_ids: items.slice(0, 4).map((c) => c.id),
    updated_at: null,
  };
  return {
    cluster,
    crops: {
      items,
      total: cropPage.total,
      page: cropPage.page,
      page_size: cropPage.page_size,
    },
  };
}

export async function getCrops(
  filter: CropFilter = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<OpCrop>> {
  type Raw = { total: number; page: number; page_size: number; crops: RawCrop[] };
  const raw = await apiFetch<Raw>(`/curation/crops${qs({ ...filter })}`, {}, signal);
  return {
    items: raw.crops.map(mapRawCrop),
    total: raw.total,
    page: raw.page,
    page_size: raw.page_size,
  };
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
  // Backend route is PUT (matches the single-crop /label PUT shape).
  return apiFetch<BulkLabelResult>(
    '/curation/crops/batch_label',
    {
      method: 'PUT',
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

/**
 * Update or clear the plate sub-bbox on a crop.
 *
 * - Pass an `[x1, y1, x2, y2]` tuple in **source-image normalized**
 *   coordinates to set/replace the plate box (server records
 *   `plate_status='human_confirmed'`).
 * - Pass `null` to clear the plate; the backend interprets this as
 *   `plate_status='no_plate_visible'`.
 *
 * Mirrors `putCropLabel` in shape. Endpoint: `PUT /curation/crops/{id}/plate`,
 * defined by backend task #32 to match this contract.
 */
export function setCropPlate(
  cropId: string,
  bbox: [number, number, number, number] | null,
  signal?: AbortSignal,
): Promise<OpCrop> {
  return apiFetch<OpCrop>(
    `/curation/crops/${encodeURIComponent(cropId)}/plate`,
    {
      method: 'PUT',
      body: JSON.stringify({ bbox_norm: bbox }),
    },
    signal,
  );
}

export async function runGemmaOnCluster(
  clusterId: number,
  signal?: AbortSignal,
): Promise<{ predicted: number; updated: number; new_class_proposals?: unknown[] }> {
  // /curation/gemma/label_batch takes {crop_ids: [...]} (max 64). Fetch the
  // unvalidated crops in this cluster first, then POST in chunks of 64.
  type CropPage = { crops: Array<{ crop_id: string }> };
  const page = await apiFetch<CropPage>(
    `/curation/crops${qs({ cluster_id: clusterId, label_validated: false, page_size: 200 })}`,
    {},
    signal,
  );
  const cropIds = page.crops.map((c) => c.crop_id);
  if (cropIds.length === 0) return { predicted: 0, updated: 0, new_class_proposals: [] };
  let predicted = 0;
  let updated = 0;
  const proposals: unknown[] = [];
  for (let i = 0; i < cropIds.length; i += 64) {
    const chunk = cropIds.slice(i, i + 64);
    const r = await apiFetch<{
      predicted: number;
      updated: number;
      new_class_proposals?: unknown[];
    }>(
      '/curation/gemma/label_batch',
      { method: 'POST', body: JSON.stringify({ crop_ids: chunk }) },
      signal,
    );
    predicted += r.predicted ?? 0;
    updated += r.updated ?? 0;
    if (Array.isArray(r.new_class_proposals)) proposals.push(...r.new_class_proposals);
  }
  return { predicted, updated, new_class_proposals: proposals };
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

export async function getReviewQueue(
  tab: ReviewTab,
  page = 1,
  pageSize = 30,
  filter: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<ReviewItem>> {
  // The /curation/review API ships bbox_norm + plate_bbox_norm as
  // [x1,y1,x2,y2] arrays. The labeler's ReviewItem extends OpCrop where
  // bboxes are {cx,cy,w,h} objects. Normalize each item through
  // mapRawCrop so PlateEditor + getThumbUrl + confirmPlate all see the
  // same shape regardless of the endpoint that produced the item.
  type RawReviewItem = RawCrop & {
    reason?: string;
    proposed_class_id?: number | null;
    proposed_class_name?: string | null;
    probe_pred_class?: string | null;
    probe_pred_entropy?: number | null;
    plate_score?: number | null;
    plate_status?: string | null;
    plate_verified?: boolean | null;
  };
  type RawPage = {
    total: number;
    page: number;
    page_size: number;
    items: RawReviewItem[];
  };
  const raw = await apiFetch<RawPage>(
    `/curation/review/${tab}${qs({ page, page_size: pageSize, ...filter })}`,
    {},
    signal,
  );
  const items: ReviewItem[] = (raw.items ?? []).map((it) => {
    const base = mapRawCrop(it);
    return {
      ...base,
      reason: it.reason ?? '',
      proposed_class_id: it.proposed_class_id ?? null,
      proposed_class_name: it.proposed_class_name ?? null,
      probe_pred_class: it.probe_pred_class ?? null,
      probe_pred_entropy: it.probe_pred_entropy ?? null,
      plate_score: it.plate_score ?? null,
      plate_status: it.plate_status ?? null,
      plate_verified: it.plate_verified ?? null,
    };
  });
  return {
    items,
    total: raw.total ?? items.length,
    page: raw.page ?? page,
    page_size: raw.page_size ?? pageSize,
  };
}

/**
 * Kick off a YOLO export job. Optional `version_tag` is included in the
 * manifest (Section L14). Server returns 202 + job id while it runs.
 */
export function exportYolo(
  opts: { version_tag?: string } = {},
  signal?: AbortSignal,
): Promise<OpExportResult> {
  const body: Record<string, unknown> = {};
  if (opts.version_tag) body.version_tag = opts.version_tag;
  return apiFetch<OpExportResult>(
    '/curation/export/yolo',
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Poll current export state. */
export function exportStatus(signal?: AbortSignal): Promise<OpExportStatus> {
  return apiFetch<OpExportStatus>('/curation/export/status', {}, signal);
}

// -- classes mutators ----------------------------------------------------

export function getClass(classId: number, signal?: AbortSignal): Promise<OpClass> {
  return apiFetch<OpClass>(`/curation/classes/${classId}`, {}, signal);
}

export function addClass(
  payload: OpClassCreate,
  signal?: AbortSignal,
): Promise<{ class_id: number; class_name: string; group: string }> {
  return apiFetch<{ class_id: number; class_name: string; group: string }>(
    '/curation/classes',
    { method: 'POST', body: JSON.stringify(payload) },
    signal,
  );
}

export function renameClass(
  classId: number,
  payload: OpClassUpdate,
  signal?: AbortSignal,
): Promise<unknown> {
  return apiFetch<unknown>(
    `/curation/classes/${classId}`,
    { method: 'PUT', body: JSON.stringify(payload) },
    signal,
  );
}

export function mergeClasses(
  payload: OpClassMerge,
  signal?: AbortSignal,
): Promise<{ source_id: number; target_id: number; relabeled: number }> {
  return apiFetch<{ source_id: number; target_id: number; relabeled: number }>(
    '/curation/classes/merge',
    { method: 'POST', body: JSON.stringify(payload) },
    signal,
  );
}

export function syncClassesToOpensearch(
  signal?: AbortSignal,
): Promise<{ created: number; updated: number }> {
  return apiFetch<{ created: number; updated: number }>(
    '/curation/classes/sync_to_opensearch',
    { method: 'POST' },
    signal,
  );
}

// -- DnD: move crops between clusters ------------------------------------

export function moveCropsToCluster(
  cropIds: string[],
  targetClusterId: number,
  signal?: AbortSignal,
): Promise<{ moved: number; failed: string[] }> {
  return apiFetch<{ moved: number; failed: string[] }>(
    '/curation/crops/move',
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds, cluster_id: targetClusterId }),
    },
    signal,
  );
}

// -- needs new class flag ------------------------------------------------

export function flagNeedsNewClass(
  cropIds: string[],
  note: string = '',
  signal?: AbortSignal,
): Promise<{ flagged: number; errors: number }> {
  return apiFetch<{ flagged: number; errors: number }>(
    '/curation/crops/flag_new_class',
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds, note }),
    },
    signal,
  );
}

// -- test holdout --------------------------------------------------------

export function freezeTestHoldout(
  payload: { percent: number; seed: number },
  signal?: AbortSignal,
): Promise<OpTestHoldoutFreezeResult> {
  return apiFetch<OpTestHoldoutFreezeResult>(
    '/curation/test_holdout/freeze',
    { method: 'POST', body: JSON.stringify(payload) },
    signal,
  );
}

export function getTestHoldoutStats(signal?: AbortSignal): Promise<OpTestHoldoutStats> {
  return apiFetch<OpTestHoldoutStats>('/curation/test_holdout/stats', {}, signal);
}

// -- registry/manifest downloads (used as anchor `download` URLs) --------

export function getClassRegistryUrl(): string {
  return `${apiBase}/curation/export/registry/class_registry.json`;
}

export function getDataYamlUrl(): string {
  return `${apiBase}/curation/export/registry/data_v7.yaml`;
}

export function getManifestUrl(): string {
  return `${apiBase}/curation/export/registry/manifest.json`;
}

// -- image URL helpers (no fetch — used directly in <img src=...>) -------

/**
 * URL for a crop thumbnail. Defaults to 160px — small enough to load
 * fast for fast grid scanning of thousands of crops, large enough that
 * vehicle details (color, body shape, headlight style) remain readable.
 * Server-side aspect-correct rendering preserves the bbox proportions.
 *
 * Pass a larger ``size`` (256-512) for click-to-inspect / focused review
 * where rendering quality matters more than transfer speed.
 */
export function getThumbUrl(cropId: string, size: number = 160): string {
  return `${apiBase}/curation/crops/${encodeURIComponent(cropId)}/thumbnail?size=${size}`;
}

export function getSourceImageUrl(cropId: string): string {
  return `${apiBase}/curation/crops/${encodeURIComponent(cropId)}/image`;
}

/**
 * Source image with bbox overlay, downscaled to ~1280px on the longest
 * side. The review page only needs the bbox to be readable, not pixel-
 * perfect — full resolution would push 2+ MB per cursor change. Callers
 * that need a pixel-accurate frame (e.g. PlateEditor) should hit
 * ``getSourceImageFull`` so the bbox lines up with the editor canvas.
 */
export function getSourceImageWithBbox(cropId: string, maxDim: number = 1280): string {
  return `${apiBase}/curation/crops/${encodeURIComponent(cropId)}/image?max_dim=${maxDim}`;
}

/** Full-resolution source image; used by PlateEditor where pixel accuracy matters. */
export function getSourceImageFull(cropId: string): string {
  return `${apiBase}/curation/crops/${encodeURIComponent(cropId)}/image`;
}

// -- training endpoints --------------------------------------------------
//
// Mirror the FastAPI `/curation/train/*` router. The labeler `/train` page is
// the only consumer; types live in `./types_train.ts` so the existing
// types.ts stays focused on the labeling data model.

/**
 * Run the preflight checks for a candidate spec WITHOUT writing
 * `job.json`. Used for inline form validation.
 *
 * The endpoint never throws on blocking issues — it returns
 * `{blocked, checks, summary}` so the UI can render every row.
 */
export function trainPreflight(
  spec: TrainJobSpec,
  signal?: AbortSignal,
): Promise<PreflightReport> {
  return apiFetch<PreflightReport>(
    '/curation/train/preflight',
    { method: 'POST', body: JSON.stringify(spec) },
    signal,
  );
}

/**
 * Start a single training run. Backend returns 422 with the preflight
 * report on blocking checks unless `force=true`. We surface those to
 * the caller as ApiError; the form reads `body.preflight` and renders.
 */
export function trainStart(
  spec: TrainJobSpec,
  force: boolean = false,
  signal?: AbortSignal,
): Promise<StartTrainResponse> {
  return apiFetch<StartTrainResponse>(
    `/curation/train/start${qs({ force: force ? true : undefined })}`,
    { method: 'POST', body: JSON.stringify(spec) },
    signal,
  );
}

/** Submit a multi-size training campaign (queued back-to-back). */
export function trainStartCampaign(
  spec: TrainCampaignSpec,
  force: boolean = false,
  signal?: AbortSignal,
): Promise<StartCampaignResponse> {
  return apiFetch<StartCampaignResponse>(
    `/curation/train/start_campaign${qs({ force: force ? true : undefined })}`,
    { method: 'POST', body: JSON.stringify(spec) },
    signal,
  );
}

/**
 * Most-recent or active job's `status.json`. Returns null when there's
 * never been a run.
 *
 * Pass a `jobId` to fetch a specific run (404 on missing).
 */
export async function getTrainStatus(
  jobId?: string,
  signal?: AbortSignal,
): Promise<TrainJobStatus | null> {
  const path = jobId
    ? `/curation/train/status/${encodeURIComponent(jobId)}`
    : '/curation/train/status';
  try {
    return await apiFetch<TrainJobStatus | null>(path, {}, signal);
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return null;
    throw e;
  }
}

export function getTrainRuns(
  limit: number = 50,
  offset: number = 0,
  signal?: AbortSignal,
): Promise<RunsListResponse> {
  return apiFetch<RunsListResponse>(
    `/curation/train/runs${qs({ limit, offset })}`,
    {},
    signal,
  );
}

export function tailTrainLog(
  jobId: string,
  lines: number = 200,
  signal?: AbortSignal,
): Promise<LogTailResponse> {
  return apiFetch<LogTailResponse>(
    `/curation/train/log/tail/${encodeURIComponent(jobId)}${qs({ lines })}`,
    {},
    signal,
  );
}

export function cancelTrainJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<CancelResponse> {
  return apiFetch<CancelResponse>(
    `/curation/train/cancel/${encodeURIComponent(jobId)}`,
    { method: 'POST' },
    signal,
  );
}

export function cancelTrainCampaign(
  campaignId: string,
  signal?: AbortSignal,
): Promise<CancelResponse> {
  return apiFetch<CancelResponse>(
    `/curation/train/cancel_campaign/${encodeURIComponent(campaignId)}`,
    { method: 'POST' },
    signal,
  );
}

export function getTrainProfiles(signal?: AbortSignal): Promise<ProfilesResponse> {
  return apiFetch<ProfilesResponse>('/curation/train/profiles', {}, signal);
}

export function getTrainPresets(signal?: AbortSignal): Promise<PresetsResponse> {
  return apiFetch<PresetsResponse>('/curation/train/presets', {}, signal);
}

export function promoteTrainJob(
  jobId: string,
  body: PromoteRequest,
  signal?: AbortSignal,
): Promise<PromoteResponse> {
  return apiFetch<PromoteResponse>(
    `/curation/train/promote/${encodeURIComponent(jobId)}`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/**
 * Run manifest (design §15.4) — full lineage envelope: dataset SHA,
 * class remap, code versions, eval results. 404 means the run finished
 * before the manifest writer was added.
 */
export function getTrainManifest(
  jobId: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  return apiFetch<Record<string, unknown>>(
    `/curation/train/manifest/${encodeURIComponent(jobId)}`,
    {},
    signal,
  );
}
