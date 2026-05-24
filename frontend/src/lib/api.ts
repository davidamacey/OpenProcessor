/**
 * Typed API client for the openprocessor `/curation/` endpoints.
 *
 * - Single base URL, defaulting to `''` (empty → relative paths, proxied
 *   by nginx in Docker production).
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
// Empty default: in Docker the labeler's nginx proxies /curation/* and /clusters/{train|assign|stats}/*
// to op-api on the same docker network. Relative URLs work from any LAN IP / VPN client.
// For local `npm run dev` outside Docker, set PUBLIC_TRITON_API_URL=http://localhost:4603 in .env.
const RAW_BASE =
  (import.meta.env?.PUBLIC_TRITON_API_URL as string | undefined) ?? '';

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

/**
 * Pipeline-dashboard payload from `GET /curation/stats/dataset`. Contract
 * defined by `src/routers/legacy/op_stats.py` — every nested key is
 * always present, numeric counters are always integers >= 0, and
 * `clusters.last_run_at` / `clusters.method` may be null when no
 * auto_label run has ever completed.
 */
export interface DatasetStats {
  as_of: string;
  total_crops: number;
  validated: number;
  test_holdout: number;
  by_source: Array<{ key: string; doc_count: number }>;
  labeled: {
    by_human: number;
    by_gemma: number;
    by_v6: number;
    by_yolo11_proposal: number;
    other: number;
  };
  plates: {
    /** Crops with a plate_bbox_norm right now — the honest "crops with a
     *  plate" count (matches the plate cluster view). */
    boxed?: number;
    /** Crops Gemma confirmed are real plates (plate_status='detected'). */
    confirmed?: number;
    /** Sum of plate_detector credit — includes rejected/failed attempts,
     *  so it OVERSTATES real plates. Kept for back-compat; not the headline. */
    total_detected: number;
    by_lpr: number;
    by_sam3: number;
    /** Legacy alias for ``by_human_drew``. */
    by_human: number;
    /** Crops where the operator drew a fresh plate bbox from scratch. */
    by_human_drew?: number;
    /** Crops whose plate was verified by a human (Confirm Plate button). */
    verified_by_human?: number;
    /** Crops whose plate was verified by Gemma (auto-verify). */
    verified_by_gemma?: number;
    /**
     * Union: any plate the operator touched — drew the bbox OR
     * confirmed an AI-proposed one. The dashboard surfaces this as
     * the honest "you reviewed N plates" number.
     */
    validated_by_human?: number;
  };
  unlabeled: {
    pending_detection: number;
    pending_verification: number;
    no_label_source: number;
  };
  in_progress: {
    sam_drain_total_unfinished: number;
  };
  clusters: {
    last_run_at: string | null;
    cluster_count: number;
    residual_count: number;
    noise_count: number;
    method: string | null;
  };
}

export function getDatasetStats(signal?: AbortSignal): Promise<DatasetStats> {
  return apiFetch<DatasetStats>('/curation/stats/dataset', {}, signal);
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
    cluster_size?: number;
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
    cluster_size: c.cluster_size ?? 0,
    added_at: c.added_at ?? '',
    color: c.color ?? null,
    deprecated: !!c.deprecated,
    hotkey_letter: c.hotkey_letter ?? null,
  }));
}

/** Raw cluster card from `/curation/clusters`. The backend is the single
 *  source of truth for every field — the frontend must never recompute
 *  dominant_class, purity, or is_unlabeled. */
type RawCluster = {
  cluster_id: number;
  cluster_kind: 'class' | 'candidate' | 'unassigned';
  size: number;
  validated_count: number;
  labelled_count: number;
  dominant_class_id: number | null;
  dominant_class_name: string | null;
  dominant_count: number;
  purity: number | null;
  is_unlabeled: boolean;
  n_subclusters: number;
  updated_at: string | null;
  representatives: Array<{
    crop_id: string;
    cluster_distance: number | null;
    class_name: string | null;
    cluster_subid: string | null;
  }>;
};

type RawClustersResp = {
  items: RawCluster[];
  total: number;
  total_class_clusters: number;
  total_candidate_clusters: number;
  cluster_id_offset: number;
};

function _rawClusterToKb(c: RawCluster): OpCluster {
  return {
    id: c.cluster_id,
    cluster_kind: c.cluster_kind,
    size: c.size,
    validated_count: c.validated_count,
    dominant_class_id: c.dominant_class_id,
    dominant_class_name: c.dominant_class_name,
    dominant_pct: c.purity,
    purity: c.purity,
    is_unlabeled: c.is_unlabeled,
    representative_crop_ids: (c.representatives ?? []).map((r) => r.crop_id),
    has_subclusters: c.n_subclusters > 0,
    n_subclusters: c.n_subclusters,
    sub_clusters: c.n_subclusters,
    updated_at: c.updated_at,
  };
}

export async function getClusters(
  filter: ClusterFilter = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<OpCluster>> {
  // Single round-trip. The backend's /curation/clusters aggregation already
  // returns dominant class, purity, validated_count, n_subclusters,
  // cluster_kind, and is_unlabeled. The frontend ONLY shapes the result
  // into the labeler's OpCluster type — no semantic compute here.
  const raw = await apiFetch<RawClustersResp>(
    `/curation/clusters${qs({
      per_cluster: 4,
      class_id: filter.class_id ?? undefined,
      // Pull enough buckets that the 512 IVF candidate clusters (+ class
      // clusters) all come back in one call — the endpoint returns them
      // ordered by size, not paginated, so a low cap would silently drop
      // the smaller candidate buckets from the "Unlabeled only" view.
      max_clusters: 2000,
      // Primary-subject grid filters — card stats reflect only passing crops.
      max_rank: filter.max_rank ?? undefined,
      min_blur_ratio: filter.min_blur_ratio ?? undefined,
      class_source: filter.class_source ?? undefined,
    })}`,
    {},
    signal,
  );
  const items = (raw.items ?? []).map(_rawClusterToKb);
  return {
    items,
    total: raw.total ?? items.length,
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
  cluster_subid?: string | null;
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
  crop_rank_in_image?: number | null;
  crop_area_norm?: number | null;
  blur_lap_ratio?: number | null;
  v6_raw_confidence?: number | null;
  coco_proposal_name?: string | null;
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
    class_source: c.class_source ?? null,
    label_source: ((c.label_source || 'model') as OpCrop['label_source']),
    label_validated: !!c.label_validated,
    label_confidence: c.confidence ?? null,
    cluster_id: c.cluster_id ?? null,
    similarity_to_centroid:
      c.cluster_distance != null ? Math.max(0, 1 - c.cluster_distance) : null,
    cluster_subid: c.cluster_subid ?? null,
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
    crop_rank_in_image: c.crop_rank_in_image ?? null,
    crop_area_norm: c.crop_area_norm ?? null,
    blur_lap_ratio: c.blur_lap_ratio ?? null,
    v6_raw_confidence: c.v6_raw_confidence ?? null,
    coco_proposal_name: c.coco_proposal_name ?? null,
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
  opts: {
    classSource?: string | null;
    maxRank?: number | null;
    minBlurRatio?: number | null;
    v6ConfLt?: number | null;
  } = {},
): Promise<{ cluster: OpCluster; crops: PaginatedResponse<OpCrop> }> {
  // Two parallel calls: paginated crops + the authoritative cluster
  // card from /curation/clusters (server-computed). The page no longer
  // derives any of the cluster's identity fields client-side.
  //
  // `classSource` narrows the crop grid to one source bucket
  // (v6_model / gemma / human / v6_low_conf / ...) without touching
  // the cluster card stats — the header still shows the whole-cluster
  // totals so the operator sees the filter against the full size.
  type CropPage = { total: number; page: number; page_size: number; crops: RawCrop[] };
  const cropQuery: Record<string, unknown> = {
    cluster_id: id,
    page,
    page_size: pageSize,
  };
  if (opts.classSource) cropQuery.class_source = opts.classSource;
  if (opts.maxRank != null) cropQuery.max_rank = opts.maxRank;
  if (opts.minBlurRatio != null) cropQuery.min_blur_ratio = opts.minBlurRatio;
  if (opts.v6ConfLt != null) cropQuery.v6_conf_lt = opts.v6ConfLt;
  const [cropPage, clustersResp] = await Promise.all([
    apiFetch<CropPage>(`/curation/crops${qs(cropQuery)}`, {}, signal),
    apiFetch<RawClustersResp>(
      // max_clusters=1 with class_id filter is the cheapest way to ask
      // for just this cluster's card.
      `/curation/clusters${qs({ per_cluster: 4, max_clusters: 1, class_id: id })}`,
      {},
      signal,
    ).catch(() => null),
  ]);
  const items = cropPage.crops.map(mapRawCrop);
  const found = clustersResp?.items?.find((c) => c.cluster_id === id) ?? null;
  const cluster: OpCluster = found
    ? _rawClusterToKb(found)
    : {
        // Fallback only if the cluster card lookup failed — leaves
        // identity fields null but lets the crop grid render.
        id,
        cluster_kind: 'unassigned',
        size: cropPage.total,
        validated_count: 0,
        dominant_class_id: null,
        dominant_class_name: null,
        dominant_pct: null,
        purity: null,
        is_unlabeled: true,
        has_subclusters: false,
        n_subclusters: 0,
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
 * Permanently dismiss a crop from every /review queue.
 *
 * Stamps ``review_dismissed_at`` + ``review_dismissed_by='human'`` on
 * the crop. The server's review_queue handler must_not's any crop with
 * ``review_dismissed_at``, so dismissed crops never reappear (until a
 * future un-dismiss endpoint is added). The crop's class / plate state
 * is left intact — only review visibility changes.
 */
export function reviewDismissCrop(
  cropId: string,
  signal?: AbortSignal,
): Promise<void> {
  return apiFetch<void>(
    `/curation/crops/${encodeURIComponent(cropId)}/review_dismiss`,
    { method: 'POST' },
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
/**
 * Fetch a single crop by id from the authoritative store. Used by the
 * review-page "Back" path so the operator sees what was actually
 * persisted rather than a possibly-stale local snapshot. Endpoint:
 * `GET /curation/crops/{crop_id}`.
 */
export async function getCrop(cropId: string, signal?: AbortSignal): Promise<OpCrop> {
  const raw = await apiFetch<RawCrop>(
    `/curation/crops/${encodeURIComponent(cropId)}`,
    {},
    signal,
  );
  return mapRawCrop(raw);
}

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

/**
 * Patch plate metadata fields without touching the bbox. Backend
 * endpoint: `PATCH /curation/crops/{id}/plate_meta`. Only the keys present in
 * `patch` are sent — pass `plate_text: null` to clear, omit to leave
 * untouched. `plate_status` must be one of `'detected' |
 * 'no_plate_visible' | 'verify_rejected' | 'false_positive'` (the
 * human-writable subset). `false_positive` keeps the detected box (for
 * FP analysis + LPR hard-negative training); `no_plate_visible` clears it.
 */
export interface PlateMetaPatch {
  plate_text?: string | null;
  plate_status?:
    | 'detected'
    | 'no_plate_visible'
    | 'verify_rejected'
    | 'false_positive'
    | null;
  plate_rejection_reason?: string | null;
}

export function updateCropPlateMeta(
  cropId: string,
  patch: PlateMetaPatch,
  signal?: AbortSignal,
): Promise<{ crop_id: string; updated_fields: string[] }> {
  return apiFetch(
    `/curation/crops/${encodeURIComponent(cropId)}/plate_meta`,
    {
      method: 'PATCH',
      body: JSON.stringify(patch),
    },
    signal,
  );
}

/**
 * Bulk-set plate_status over many crops. Backend: `POST /curation/plates/batch_status`.
 * The cluster-view triage op: select outlier plates → mark all false_positive,
 * or bulk-confirm good plates (status='detected' + plateVerified=true).
 */
export function batchPlateStatus(
  cropIds: string[],
  plateStatus: 'detected' | 'no_plate_visible' | 'verify_rejected' | 'false_positive',
  opts: { plateVerified?: boolean; labelSource?: string } = {},
  signal?: AbortSignal,
): Promise<{ updated: number; conflicts: { crop_id: string; current_source: string | null }[] }> {
  return apiFetch(
    '/curation/plates/batch_status',
    {
      method: 'POST',
      body: JSON.stringify({
        crop_ids: cropIds,
        plate_status: plateStatus,
        plate_verified: opts.plateVerified ?? null,
        label_source: opts.labelSource ?? 'human',
      }),
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

export type RefineClusterResponse = {
  cluster_id: number;
  n_members: number;
  n_subclusters: number;
  purity: number;
  subcluster_weighted_purity?: number;
  n_updated?: number;
  distance_threshold?: number;
  linkage?: string;
  metric?: string;
  action: 'refined' | 'skipped_too_small' | 'skipped_too_large';
  reason?: string;
};

export function refineCluster(
  clusterId: number,
  signal?: AbortSignal,
): Promise<RefineClusterResponse> {
  return apiFetch<RefineClusterResponse>(
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
): Promise<BulkLabelResult> {
  // Backend returns the same {updated, conflicts: [...]} shape as
  // /curation/crops/batch_label. Reuse the type so both call sites share the
  // conflict-handling code path.
  return apiFetch<BulkLabelResult>(
    '/curation/crops/move',
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds, cluster_id: targetClusterId }),
    },
    signal,
  );
}

// -- exclude / ignore crops ----------------------------------------------

/** Reasons the operator can attach when ignoring crops. 'ignore' is the
 *  default one-click value; the others record *why* for later analysis. */
export type ExcludeReason =
  | 'ignore'
  | 'blurry'
  | 'unidentifiable'
  | 'not_a_vehicle'
  | 'partial_crop';

export function excludeCrops(
  cropIds: string[],
  reason: ExcludeReason = 'ignore',
  signal?: AbortSignal,
): Promise<{ excluded: number; errors: number }> {
  return apiFetch<{ excluded: number; errors: number }>(
    '/curation/crops/batch_exclude',
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds, reason }),
    },
    signal,
  );
}

export function unexcludeCrops(
  cropIds: string[],
  signal?: AbortSignal,
): Promise<{ unexcluded: number; errors: number }> {
  return apiFetch<{ unexcluded: number; errors: number }>(
    '/curation/crops/batch_unexclude',
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds }),
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
export function getSourceImageWithBbox(
  cropId: string,
  maxDim: number = 1280,
  cacheKey?: string | null,
): string {
  // The server reads plate_bbox_norm from OpenSearch and burns the
  // overlay into the JPEG. The crop_id alone produces an identical URL
  // across edits, so the browser cache returns the pre-edit JPEG and
  // the left-side preview lags the right-side canvas. Pass a key that
  // changes when the bbox changes (e.g. the bbox tuple) to bust the
  // cache on edits while still hitting the cache between cursor moves.
  const base = `${apiBase}/curation/crops/${encodeURIComponent(cropId)}/image?max_dim=${maxDim}`;
  return cacheKey ? `${base}&v=${encodeURIComponent(cacheKey)}` : base;
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

// -- Auto-label (recluster) job ------------------------------------------
// Wraps POST /curation/pipeline/auto_label/{start,status,cancel}. The pipeline
// re-runs prototype assignment → cluster_id normalize → AHC residuals → auto-
// promote → Gemma sweep, fixing prototype drift and stale cluster_id on
// labeled crops. Hours at HDD scale; the panel polls status while it runs.

export interface AutoLabelStartParams {
  train_clusters?: boolean;
  promote_min_purity?: number;
  promote_min_members?: number;
  gemma_batch_size?: number;
  gemma_concurrency?: number;
  max_gemma_crops?: number;
  v6_confidence_skip_gemma?: number;
  /** When true, broaden the residual AHC pool to include items already
   *  in candidate clusters so smaller candidates can merge into bigger
   *  ones. Default false — only fresh / class-bucketed items are
   *  re-clustered. */
  recluster_unvalidated?: boolean;
  // -- Cluster scope (primary-subject gate) ------------------------------
  /** Train + assign only crops with crop_rank_in_image <= this (1 = largest,
   *  2 = largest + 2nd). Smaller crops are parked. Needs full rank/blur
   *  backfill; the run blocks otherwise. */
  gate_max_rank?: number | null;
  /** Train + assign only crops with blur_lap_ratio >= this. */
  gate_min_blur_ratio?: number | null;
  /** IVF fixed centroid count (default 512). Sweep down with the gate on. */
  n_clusters?: number | null;
}

export type AutoLabelStatus =
  | 'idle'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface AutoLabelJobState {
  job_id: string;
  status: AutoLabelStatus;
  stage: string;
  processed: number;
  total: number;
  started_at: number;
  finished_at: number;
  error: string | null;
  result: Record<string, unknown>;
  args: Record<string, unknown>;
  eta_seconds: number | null;
  elapsed_seconds: number;
  // GPU clustering telemetry (optional — older worker versions omit).
  // `backend` is 'gpu' when cluster_residuals' UMAP ran on cuML;
  // 'cpu' for sklearn fallback. `backend_detail` is a human-readable
  // chip ("gpu (cuml 26.4 · A6000 GPU 0 · 9824/49152 MB free · nn_descent)"
  // or "cpu (sklearn 1.4 · umap-learn 0.5)").
  backend?: 'gpu' | 'cpu' | null;
  backend_detail?: string | null;
  free_vram_mb?: number | null;
  peak_vram_mb?: number | null;
  stage_durations?: Record<string, number>;
}

export function startAutoLabel(
  params: AutoLabelStartParams = {},
  signal?: AbortSignal,
): Promise<AutoLabelJobState> {
  return apiFetch<AutoLabelJobState>(
    `/curation/pipeline/auto_label/start${qs(params as Record<string, unknown>)}`,
    { method: 'POST' },
    signal,
  );
}

export function getAutoLabelStatus(signal?: AbortSignal): Promise<AutoLabelJobState> {
  return apiFetch<AutoLabelJobState>(
    '/curation/pipeline/auto_label/status',
    {},
    signal,
  );
}

export function cancelAutoLabel(
  signal?: AbortSignal,
): Promise<AutoLabelJobState & { cancelled: boolean }> {
  return apiFetch<AutoLabelJobState & { cancelled: boolean }>(
    '/curation/pipeline/auto_label/cancel',
    { method: 'POST' },
    signal,
  );
}
