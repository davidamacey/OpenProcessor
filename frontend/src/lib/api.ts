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

import {
  FALLBACK_METHODS,
  parseKbMethodsResponse,
  type OpMethodsResponse,
} from './strategies';
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
  OpDatasetList,
  OpExportResult,
  OpExportStatus,
  OpHealth,
  OpLprExportResult,
  OpLprExportStatus,
  OpModelsStatus,
  OpStats,
  OpTestHoldoutFreezeResult,
  OpTestHoldoutStats,
  DiverseSelection,
  PaginatedResponse,
  ReviewItem,
  ReviewTab,
  SearchCrop,
  SelectDiverseScope,
  SelectJobStatus,
  UnloadModelResponse,
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
const RAW_BASE = (import.meta.env?.PUBLIC_TRITON_API_URL as string | undefined) ?? '';

export const apiBase: string = RAW_BASE.replace(/\/+$/, '');

const DETAIL_MAX_CHARS = 200;

/**
 * Pull the human-readable reason out of an error response body.
 *
 * The backend is FastAPI, so 4xx bodies are `{detail: "..."}` (occasionally
 * `{message: "..."}`, or a plain-text body). Without this, every toast in
 * the app shows `API 422 http://…/batch_label` and the operator has no idea
 * what the server objected to — the callsites all render `Error.message`.
 */
function errorDetail(body: unknown): string | null {
  let raw: unknown = null;
  if (typeof body === 'string') {
    raw = body;
  } else if (body && typeof body === 'object') {
    const rec = body as Record<string, unknown>;
    raw = rec.detail ?? rec.message ?? null;
  }
  if (typeof raw !== 'string') return null;
  const text = raw.trim();
  if (!text) return null;
  return text.length > DETAIL_MAX_CHARS
    ? `${text.slice(0, DETAIL_MAX_CHARS - 1)}…`
    : text;
}

export class ApiError extends Error {
  status: number;
  body: unknown;
  url: string;
  /** The server's `detail`/`message` string, when it sent one. */
  detail: string | null;
  constructor(status: number, url: string, body: unknown, message?: string) {
    const detail = errorDetail(body);
    super(message ?? `API ${status} ${url}${detail ? ` — ${detail}` : ''}`);
    this.status = status;
    this.body = body;
    this.url = url;
    this.detail = detail;
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

/**
 * Resolve a path/URL against the configured `apiBase`. Idempotent — an
 * already-absolute URL (or one already prefixed with `apiBase`) passes
 * through unchanged, so it's safe to call on a value that might have
 * already been resolved upstream (e.g. a server-supplied
 * `representative_thumb_urls` entry rendered through a shared helper).
 *
 * The backend intentionally emits relative `/curation/...` URLs in API
 * response bodies (e.g. `plate_thumbnail_url`, `representative_thumb_urls`)
 * so the same payload works both same-origin (production nginx proxy,
 * empty `apiBase`) and cross-origin (a remote `PUBLIC_TRITON_API_URL`).
 * Resolving those against `apiBase` is the client's job — every render
 * site that puts a server-supplied URL into an `<img src>` must go
 * through this function first.
 */
export function resolveApiUrl(url: string): string {
  if (url.startsWith('http')) return url;
  if (apiBase && url.startsWith(apiBase)) return url;
  return `${apiBase}${url}`;
}

export async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const url = resolveApiUrl(path);
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
      // An abort during the backoff propagates — the caller cancelled.
      await sleep(RETRY_DELAYS_MS[attempt]!, signal);
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

/**
 * Capability discovery for the curation-strategy registries (plan §3/§5.3):
 * which cluster methods / review sorts / overlays / scores the backend
 * currently offers, each with a `stable | experimental | shadow |
 * disabled` status. Phase 0 plumbing only — nothing consumes this yet.
 *
 * **Never rejects.** `/curation/methods` may not exist yet (backend Phase 0
 * lands independently — see `strategies.ts`'s header), and this endpoint
 * is pure capability discovery, not something a caller should have to
 * try/catch around. `apiFetch` already applies the house retry rule (no
 * retry on 4xx, 3 retries with backoff on 5xx/network errors); once that
 * settles, a 404 or any other failure here resolves to `FALLBACK_METHODS`
 * — the hardcoded stable-only list matching what's actually implemented
 * today — instead of throwing. A caller-initiated abort still propagates,
 * since that's a cancellation, not a backend failure.
 */
export async function getMethods(signal?: AbortSignal): Promise<OpMethodsResponse> {
  try {
    const raw = await apiFetch<unknown>('/curation/methods', {}, signal);
    return parseKbMethodsResponse(raw);
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    return FALLBACK_METHODS;
  }
}

// -- embedding projection (2-d visualization overlay, Phase 5) -----------
//
// docs/curation-strategy-plan-2026-09.md §2.7/§5.6/§7 — `embedding_viz.py`
// + `op_viz.py` (openprocessor). UMAP-as-a-visualization-only overlay is the
// one method in the whole curation-strategy plan that was NOT validated
// in Phase 2 before implementation started; its own §6 acceptance bar
// (2-d neighborhood purity vs. the real IVF cluster_id) decides whether
// it ships plain, ships behind an "approximate" banner, or doesn't ship
// at all. Nothing here assumes an outcome — `isEmbeddingVizAvailable`
// (strategies.ts) is what actually gates whether any UI renders at all,
// exactly like `isDiverseOverlayAvailable` gates Phase 4's `diverse`
// overlay.
//
// Coordinates are cached/batch-computed server-side (plan §2.7's
// non-negotiable rule: "never fit on a request path") — `getVizProjection`
// only ever serves whatever the last `rebuildVizProjection()` job
// produced.

/**
 * One projected point. Color is derived client-side from `cluster_id`
 * (see `colorForCluster` in `embeddingPlot.ts`) — this overlay decorates
 * an existing assignment, it never computes or chooses one.
 */
export interface VizPoint {
  crop_id: string;
  x: number;
  y: number;
  cluster_id: number | null;
  class_name: string | null;
  class_source: string | null;
}

export interface VizProjectionResponse {
  points: VizPoint[];
  /** `points.length`, kept as a field for parity with other list responses
   *  in this file — the server doesn't send a separate total, there's no
   *  pagination here (`max_points` is a hard cap, not a page size). */
  total: number;
  /**
   * CONFIRMED (2026-09-10) against the real `GET /curation/viz/projection`
   * (`embedding_viz.get_cached_projection`): the server returns
   * `{status: 'not_built'}` when nothing has been fit yet, or
   * `{points, projection_version, fitted_at, stale}` otherwise — there is
   * no `built`/`not_built` boolean on the wire. `built` here is this
   * file's own derived convenience (`status !== 'not_built'`), kept so
   * `EmbeddingPlot` doesn't need to know the raw sentinel shape.
   */
  built: boolean;
  fitted_at: string | null;
  projection_version: string | null;
  /** True iff some in-scope crop's cached coordinates predate the latest
   *  fit (partial-coverage signal, same philosophy as the scores
   *  endpoints' `field_coverage`) — `false` when `built` is `false`. */
  stale: boolean;
}

const EMPTY_VIZ_PROJECTION: VizProjectionResponse = {
  points: [],
  total: 0,
  built: false,
  fitted_at: null,
  projection_version: null,
  stale: false,
};

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

function parseVizPoint(raw: unknown): VizPoint | null {
  if (!isPlainObject(raw)) return null;
  const { crop_id, x, y } = raw;
  if (typeof crop_id !== 'string' || !crop_id) return null;
  if (typeof x !== 'number' || !Number.isFinite(x)) return null;
  if (typeof y !== 'number' || !Number.isFinite(y)) return null;
  const cluster_id =
    typeof raw.cluster_id === 'number' && Number.isFinite(raw.cluster_id)
      ? raw.cluster_id
      : null;
  return {
    crop_id,
    x,
    y,
    cluster_id,
    class_name: typeof raw.class_name === 'string' ? raw.class_name : null,
    class_source: typeof raw.class_source === 'string' ? raw.class_source : null,
  };
}

/**
 * Fetch the cached 2-d projection. **Never rejects** (mirrors
 * `getMethods`'s contract) — `/curation/viz/projection` may not exist yet (the
 * backend Phase 5 lands independently of this frontend branch) or may
 * 404/5xx for any other reason, and a fetch failure here should degrade
 * `EmbeddingPlot` to its pending/empty state rather than crash the page
 * it replaced the grid on. A caller-initiated abort still propagates —
 * that's a cancellation, not a backend failure.
 *
 * The real payload is `{status: 'not_built'}` (nothing fit yet) or
 * `{points, projection_version, fitted_at, stale}` (confirmed 2026-09-10
 * against `embedding_viz.get_cached_projection`) — normalized here into
 * this file's own `built`/`fitted_at`/`projection_version`/`stale` shape.
 */
export async function getVizProjection(
  params: {
    cluster_id?: number | null;
    class_id?: number | null;
    max_points?: number;
  } = {},
  signal?: AbortSignal,
): Promise<VizProjectionResponse> {
  try {
    const raw = await apiFetch<unknown>(
      `/curation/viz/projection${qs({
        cluster_id: params.cluster_id ?? undefined,
        class_id: params.class_id ?? undefined,
        max_points: params.max_points ?? undefined,
      })}`,
      {},
      signal,
    );
    if (!isPlainObject(raw)) return EMPTY_VIZ_PROJECTION;
    if (raw.status === 'not_built') return EMPTY_VIZ_PROJECTION;
    const rawPoints = Array.isArray(raw.points) ? raw.points : [];
    const points = rawPoints.map(parseVizPoint).filter((p): p is VizPoint => p != null);
    return {
      points,
      total: points.length,
      built: true,
      fitted_at: typeof raw.fitted_at === 'string' ? raw.fitted_at : null,
      projection_version:
        typeof raw.projection_version === 'string' ? raw.projection_version : null,
      stale: raw.stale === true,
    };
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    return EMPTY_VIZ_PROJECTION;
  }
}

/**
 * Background rebuild-job snapshot. CONFIRMED (2026-09-10) against the
 * real `embedding_viz._JobState` — flat, not the nested
 * `{running, result: {...}}` shape this file originally guessed:
 * `status` is a string enum (`'idle' | 'running' | 'completed' |
 * 'failed' | 'cancelled'`), timestamps are unix-epoch numbers (`0` when
 * unset, not `null`), and `n_written`/`projection_version` are top-level
 * fields, not nested under a `result` key.
 */
export interface VizProjectionJob {
  job_id: string;
  status: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';
  scope: string;
  cluster_id: number | null;
  n_pool: number;
  n_written: number;
  started_at: number;
  finished_at: number;
  error: string | null;
  projection_version: string | null;
}

/**
 * Kick off a projection (re)fit — a background job, not a request-path
 * fit (plan §2.7/§3: "coordinates cached/batch-computed... never fit on
 * a request path"). Unlike `getVizProjection`, this is an explicit
 * user-initiated action (the operator clicked "Rebuild"), so — matching
 * the `buildPlateFpCentroids`/`clusterPlates` precedent — it lets the
 * error propagate for the caller to catch + toast rather than swallowing
 * it into a fallback value.
 */
export function rebuildVizProjection(signal?: AbortSignal): Promise<VizProjectionJob> {
  return apiFetch<VizProjectionJob>(
    '/curation/viz/projection/rebuild',
    { method: 'POST' },
    signal,
  );
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
  plate_validated: boolean | null;
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
  /** Parent-crop rank by size in its image (1 = largest). */
  crop_rank_in_image?: number | null;
  crop_area_norm?: number | null;
  /** Plate clustering assignment (independent of vehicle cluster_id). */
  plate_cluster_id?: number | null;
  plate_cluster_subid?: string | null;
  plate_cluster_distance?: number | null;
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
  /** Plate clustering bucket (independent of the vehicle cluster_id). */
  plate_cluster_id?: number;
  /** AHC plate sub-cluster id (e.g. "17a"). */
  plate_cluster_subid?: string;
  /** Order a bucket's plates by sub-cluster so AHC groups come back contiguous. */
  sort_by_subid?: boolean;
  /** Only plates on the top-N largest crops (crop_rank_in_image<=N). */
  max_rank?: number;
  min_score?: number;
  max_score?: number;
  verified?: boolean;
  detector?: string;
  text?: string;
  include_test?: boolean;
}

export function getPlates(
  params: PlatesQuery = {},
  signal?: AbortSignal,
): Promise<PlatesPage> {
  return apiFetch<PlatesPage>(
    `/curation/plates${qs(params as Record<string, unknown>)}`,
    {},
    signal,
  );
}

/** Plate-clustering background-job snapshot. The one-click pipeline result also
 *  carries the FP-rebuild + auto-assign sub-steps, and may report a re-partition
 *  that was skipped to protect a fresh manual refine (TTL). */
export interface PlateClusterJob {
  running: boolean;
  started_at: string | null;
  finished_at: string | null;
  result:
    | ({
        status?: string;
        n_plates?: number;
        n_clusters?: number;
        assigned?: number;
        fp_centroids?: { status: string; n_members?: number; k?: number };
        auto_fp?: { status: string; n_moved?: number; threshold?: number };
      } & Record<string, unknown>)
    | null;
  error: string | null;
}

/** Launch the one-click plate-clustering pipeline (background job — 50k+ plates
 *  take minutes): rebuild FP sub-centroids → auto-pull tight FP matches → re-partition
 *  the good plates. Returns immediately; poll getPlateClusterStatus for completion. */
export function clusterPlates(
  maxRank?: number,
  opts: { forceRepartition?: boolean; autoFpThreshold?: number } = {},
  signal?: AbortSignal,
): Promise<PlateClusterJob> {
  return apiFetch(
    `/curation/plates/cluster${qs({
      max_rank: maxRank,
      force_repartition: opts.forceRepartition,
      auto_fp_threshold: opts.autoFpThreshold,
    })}`,
    { method: 'POST' },
    signal,
  );
}

/** Poll the background plate-clustering job. */
export function getPlateClusterStatus(signal?: AbortSignal): Promise<PlateClusterJob> {
  return apiFetch('/curation/plates/cluster/status', {}, signal);
}

/** Per-bucket AHC refine over plate_pe_embedding; writes plate_cluster_subid. */
export function refinePlateCluster(
  clusterId: number,
  signal?: AbortSignal,
): Promise<{
  cluster_id: number;
  n_members: number;
  n_subclusters: number;
  action: string;
}> {
  return apiFetch(`/curation/plates/clusters/refine/${clusterId}`, { method: 'POST' }, signal);
}

/** Plate cluster cards (mirrors getClusters' OpCluster shape). */
export function getPlateClusters(
  opts: { maxClusters?: number; perCluster?: number; maxRank?: number } = {},
  signal?: AbortSignal,
): Promise<{ clusters: OpCluster[]; count: number }> {
  return apiFetch(
    `/curation/plates/clusters${qs({
      max_clusters: opts.maxClusters,
      per_cluster: opts.perCluster,
      max_rank: opts.maxRank,
    })}`,
    {},
    signal,
  );
}

/** Background FP-centroid build-job snapshot + persisted centroid metadata. */
export interface PlateFpCentroidJob {
  running: boolean;
  started_at: string | null;
  finished_at: string | null;
  result: { status: string; n_members: number; k: number } | null;
  error: string | null;
  centroids: {
    trained_at: string | null;
    k: number | null;
    n_members: number | null;
  } | null;
}

/** (Re)build the FP centroid store — sub-types the FP bucket (background job). */
export function buildPlateFpCentroids(signal?: AbortSignal): Promise<PlateFpCentroidJob> {
  return apiFetch('/curation/plates/fp_centroids/build', { method: 'POST' }, signal);
}

/** Poll the FP-centroid build job + read persisted centroid metadata. */
export function getPlateFpCentroidStatus(
  signal?: AbortSignal,
): Promise<PlateFpCentroidJob> {
  return apiFetch('/curation/plates/fp_centroids/status', {}, signal);
}

export interface SuspectedFpItem extends PlateBrowseItem {
  suspected_fp_distance: number;
  nearest_fp_subid: string | null;
}

export interface SuspectedFpPage {
  items: SuspectedFpItem[];
  total: number;
  page: number;
  page_size: number;
  threshold?: number;
  centroids_built: boolean;
  trained_at?: string | null;
  message?: string;
}

/** Non-FP plate crops ranked by similarity to the known FP centroids. */
export function getSuspectedFalsePositives(
  opts: { threshold?: number; page?: number; pageSize?: number } = {},
  signal?: AbortSignal,
): Promise<SuspectedFpPage> {
  return apiFetch(
    `/curation/plates/suspected_false_positives${qs({
      threshold: opts.threshold,
      page: opts.page,
      page_size: opts.pageSize,
    })}`,
    {},
    signal,
  );
}

export type TrainingCohortMode =
  | 'lpr_blind_spots'
  | 'lpr_low_conf_correct'
  | 'disagreement'
  | 'human_corrected'
  | 'false_positives';

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
 * Unload a Triton model and remove its repo directory (follow-up gap 2,
 * docs/design/audit-remediation-plan-2026-09.md Appendix D item 3,
 * 2026-09-11). The backend enforces the real guard (LPR models never,
 * active/core models need `force`) — `force` here only matters for the
 * latter; passing it for an LPR model still 403s.
 */
export function unloadModel(
  modelName: string,
  force = false,
  signal?: AbortSignal,
): Promise<UnloadModelResponse> {
  return apiFetch<UnloadModelResponse>(
    `/curation/models/${encodeURIComponent(modelName)}${qs({ force })}`,
    { method: 'DELETE' },
    signal,
  );
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
  const totalImages = (ds.by_source ?? []).reduce(
    (acc, b) => acc + (b.doc_count || 0),
    0,
  );
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
  const res = await apiFetch<{ classes: RawClass[] } | RawClass[]>(
    '/curation/classes',
    {},
    signal,
  );
  const raw = Array.isArray(res) ? res : (res.classes ?? []);
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
  // Curation scores (Phase 3, docs/curation-strategy-plan-2026-09.md §4).
  // Optional/forward-tolerant: an un-backfilled pool just omits these.
  mistakenness_score?: number | null;
  mistakenness_method?: string | null;
  mistakenness_version?: string | null;
  mistakenness_scored_at?: string | null;
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
    label_source: (c.label_source || 'model') as OpCrop['label_source'],
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
    mistakenness_score: c.mistakenness_score ?? null,
    mistakenness_method: c.mistakenness_method ?? null,
    mistakenness_version: c.mistakenness_version ?? null,
    mistakenness_scored_at: c.mistakenness_scored_at ?? null,
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
    /**
     * Forwarded verbatim to `/curation/crops?order=`. Only `'outliers'` is
     * special-cased server-side today (op_crops.py `order` query param —
     * see docs/curation-strategy-plan-2026-09.md §1); an id the backend
     * doesn't recognize is harmless (qs() still sends it, the server
     * just falls back to its default ordering). Typed as `string` rather
     * than a fixed union so a new `/curation/methods`-reported order id doesn't
     * require touching this signature — callers should still gate which
     * ids they actually offer against what `/curation/methods` reports.
     */
    order?: string | null;
    /**
     * Pool-scale overlay parameter, forwarded to `/curation/crops?k=` only when
     * set (curation-strategy plan Phase 4 — `order: 'diverse'`'s "how
     * many diverse crops" count). Meaningless for every other `order`
     * value; the caller (`/clusters/[id]`) only sets it in diverse mode.
     */
    k?: number | null;
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
  type CropPage = {
    total: number;
    page: number;
    page_size: number;
    crops: RawCrop[];
    /** Only present for a pool-scale overlay ordering (e.g.
     *  `order=diverse`) — see `PaginatedResponse.order_method` /
     *  `.order_version` / `.n_pool` in types.ts. Untyped/optional and
     *  parsed tolerantly below: the shape belongs to whichever `order`
     *  overlay is active, not something this function should assume. */
    method?: unknown;
    version?: unknown;
    n_pool?: unknown;
  };
  const cropQuery: Record<string, unknown> = {
    cluster_id: id,
    page,
    page_size: pageSize,
  };
  if (opts.classSource) cropQuery.class_source = opts.classSource;
  if (opts.maxRank != null) cropQuery.max_rank = opts.maxRank;
  if (opts.minBlurRatio != null) cropQuery.min_blur_ratio = opts.minBlurRatio;
  if (opts.v6ConfLt != null) cropQuery.v6_conf_lt = opts.v6ConfLt;
  if (opts.order) cropQuery.order = opts.order;
  if (opts.k != null) cropQuery.k = opts.k;
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
      order_method: typeof cropPage.method === 'string' ? cropPage.method : null,
      order_version: typeof cropPage.version === 'string' ? cropPage.version : null,
      n_pool: typeof cropPage.n_pool === 'number' ? cropPage.n_pool : null,
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
export function reviewDismissCrop(cropId: string, signal?: AbortSignal): Promise<void> {
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
): Promise<{
  updated: number;
  conflicts: { crop_id: string; current_source: string | null }[];
}> {
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
    /** Set when the requested `?sort=` fell back to the default — see
     *  PaginatedResponse.sort_fallback_reason in types.ts. */
    sort_fallback_reason?: string | null;
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
    sort_fallback_reason: raw.sort_fallback_reason ?? null,
  };
}

// -- pool-scale diverse selection for /review (P2-10) --------------------
//
// `POST /curation/select/diverse` is a DIFFERENT contract from `/clusters/[id]`'s
// `GET /curation/crops?order=diverse&k=N`: that path is a small, synchronous,
// cluster-scoped selection; this one scopes to a review-tab cohort that
// can be pool-scale (the `all` tab is ~320k crops), so the backend may
// answer either 200 (small pool, `crop_ids` ready now) or 202 (large pool,
// `job_id` — poll `getSelectStatus`/cancel via `cancelSelect`). This is a
// singleton job server-side: a second POST while one is running 409s.

/** Discriminated result of `selectDiverse` — never throws for the two
 *  expected non-2xx states (`disabled` on a 400 "feature off" response,
 *  `already_running` on a 409 singleton-job collision). Any other error
 *  (network, 5xx after retries, unexpected 4xx) still propagates as an
 *  ApiError/DOMException, same as every other endpoint in this file —
 *  those are real failures, not states the UI should treat as routine. */
export type SelectDiverseResult =
  | { kind: 'ready'; selection: DiverseSelection }
  | { kind: 'job'; job_id: string }
  | { kind: 'disabled' }
  | { kind: 'already_running' };

function parseDiverseSelection(raw: unknown): DiverseSelection | null {
  if (!isPlainObject(raw)) return null;
  const { crop_ids, n_pool } = raw;
  if (!Array.isArray(crop_ids)) return null;
  if (typeof n_pool !== 'number') return null;
  return {
    crop_ids: crop_ids.filter((id): id is string => typeof id === 'string'),
    method: typeof raw.method === 'string' ? raw.method : 'diverse',
    version: typeof raw.version === 'string' ? raw.version : '',
    n_pool,
  };
}

/**
 * Run (or resume) a pool-scale diverse selection. Distinguishes the 200
 * ("ready now", small pool) vs. 202 ("job enqueued", large pool — the
 * `all` tab will always take this path) response shapes by which fields
 * are present in the parsed JSON body, since `apiFetch` only exposes the
 * parsed body, not the raw `Response`/status, for a 2xx result.
 */
export async function selectDiverse(
  scope: SelectDiverseScope,
  k: number,
  seedCropId?: string | null,
  signal?: AbortSignal,
): Promise<SelectDiverseResult> {
  try {
    const raw = await apiFetch<unknown>(
      '/curation/select/diverse',
      {
        method: 'POST',
        body: JSON.stringify({
          scope,
          k,
          ...(seedCropId ? { seed_crop_id: seedCropId } : {}),
        }),
      },
      signal,
    );
    if (isPlainObject(raw) && typeof raw.job_id === 'string') {
      return { kind: 'job', job_id: raw.job_id };
    }
    const selection = parseDiverseSelection(raw);
    if (selection) return { kind: 'ready', selection };
    // Unrecognized 2xx shape — treat as an empty-but-valid selection
    // rather than throwing, mirroring this file's forward-tolerant
    // convention for capability-discovery-adjacent endpoints.
    return { kind: 'ready', selection: { crop_ids: [], method: 'diverse', version: '', n_pool: 0 } };
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    if (e instanceof ApiError && e.status === 400) return { kind: 'disabled' };
    if (e instanceof ApiError && e.status === 409) return { kind: 'already_running' };
    throw e;
  }
}

function parseSelectJobStatus(raw: unknown): SelectJobStatus {
  if (!isPlainObject(raw)) return { status: 'unknown' };
  return {
    job_id: typeof raw.job_id === 'string' ? raw.job_id : null,
    status: typeof raw.status === 'string' ? raw.status : 'unknown',
    // `result` is only populated once status === 'completed' (job.py's
    // _JobState) — null/absent every other status, including 'failed'/
    // 'cancelled', where there's nothing to hydrate.
    result: parseDiverseSelection(raw.result),
    error: typeof raw.error === 'string' ? raw.error : null,
  };
}

/** Poll the singleton diverse-selection job. Caller decides polling
 *  cadence/cleanup (see `/review`'s `+page.svelte` — mirrors the
 *  `bakeoff` page's setInterval/clearInterval pattern). */
export async function getSelectStatus(signal?: AbortSignal): Promise<SelectJobStatus> {
  const raw = await apiFetch<unknown>('/curation/select/status', {}, signal);
  return parseSelectJobStatus(raw);
}

/** Cancel the singleton diverse-selection job, if any is running. Real
 *  backend returns `{cancelled: bool, ...job state}` (op_select.py's
 *  `select_cancel`), not a bare 204 — the caller only needs to know
 *  polling can stop, so the body is discarded. */
export async function cancelSelect(signal?: AbortSignal): Promise<void> {
  await apiFetch<unknown>('/curation/select/cancel', { method: 'POST' }, signal);
}

/**
 * Free-text semantic search over vehicle crops (P2-14). Backend:
 * `GET /curation/search/text`, gated behind the `semantic_search` overlay in
 * `/curation/methods` (see `isSemanticSearchAvailable` in `./strategies`) — a
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
  type RawSearchItem = RawCrop & { similarity_score?: number | null; score?: number | null };
  type RawPage = {
    total: number;
    page: number;
    page_size: number;
    items: RawSearchItem[];
  };
  const raw = await apiFetch<RawPage>(
    `/curation/search/text${qs({ q, page, page_size: pageSize, ...filter })}`,
    {},
    signal,
  );
  const items: SearchCrop[] = (raw.items ?? []).map((it) => {
    const base = mapRawCrop(it);
    return {
      ...base,
      similarity_score: it.similarity_score ?? it.score ?? 0,
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

/**
 * Build a standalone single-class LPR (license-plate) YOLO dataset.
 * Synchronous on the server; returns the export dir + counts when done.
 */
export function exportLpr(
  opts: {
    version_tag?: string;
    empty_bg_ratio?: number;
    max_positive_images?: number;
    skip_test_split?: boolean;
    dedup_threshold?: number | null;
    image_mode?: 'whole_frame' | 'vehicle_crop';
    img_max_side?: 640 | 1280;
  } = {},
  signal?: AbortSignal,
): Promise<OpLprExportResult> {
  const body: Record<string, unknown> = {};
  if (opts.version_tag) body.version_tag = opts.version_tag;
  if (opts.empty_bg_ratio !== undefined) body.empty_bg_ratio = opts.empty_bg_ratio;
  if (opts.max_positive_images !== undefined)
    body.max_positive_images = opts.max_positive_images;
  if (opts.skip_test_split !== undefined) body.skip_test_split = opts.skip_test_split;
  if (opts.dedup_threshold !== undefined) body.dedup_threshold = opts.dedup_threshold;
  if (opts.image_mode !== undefined) body.image_mode = opts.image_mode;
  if (opts.img_max_side !== undefined) body.img_max_side = opts.img_max_side;
  return apiFetch<OpLprExportResult>(
    '/curation/export/lpr',
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Last LPR-export status (reads the LPR `current` symlink + manifest). */
export function exportLprStatus(signal?: AbortSignal): Promise<OpLprExportStatus> {
  return apiFetch<OpLprExportStatus>('/curation/export/lpr/status', {}, signal);
}

/**
 * List every materialized dataset version on disk (newest first) so the
 * operator can train on any past export — a small sample, a larger subset, or
 * the full set — for consistent data re-use across model-size upgrades.
 */
export function listDatasets(
  kind?: 'lpr' | 'vehicles',
  signal?: AbortSignal,
): Promise<OpDatasetList> {
  const qs = kind ? `?kind=${encodeURIComponent(kind)}` : '';
  return apiFetch<OpDatasetList>(`/curation/export/datasets${qs}`, {}, signal);
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
): Promise<{
  source_id: number;
  target_id: number;
  deprecated: boolean;
  source_name: string;
  target_name: string;
}> {
  // The backend does not return a relabeled count -- don't claim one.
  return apiFetch<{
    source_id: number;
    target_id: number;
    deprecated: boolean;
    source_name: string;
    target_name: string;
  }>('/curation/classes/merge', { method: 'POST', body: JSON.stringify(payload) }, signal);
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
  return `${apiBase}/curation/export/registry/data.yaml`;
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

/**
 * URL for a plate close-up thumbnail (the plate sub-bbox rendered to a
 * tile), same construction convention as {@link getThumbUrl}. Pass
 * `cacheBustKey` (e.g. `Date.now()`) after a bbox edit so the browser
 * doesn't serve the pre-edit crop from its image cache.
 */
export function getPlateThumbUrl(
  cropId: string,
  size: number = 160,
  cacheBustKey?: string | number | null,
): string {
  const base = `${apiBase}/curation/crops/${encodeURIComponent(cropId)}/plate_thumbnail?size=${size}`;
  return cacheBustKey != null ? `${base}&v=${encodeURIComponent(cacheBustKey)}` : base;
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
  return apiFetch<RunsListResponse>(`/curation/train/runs${qs({ limit, offset })}`, {}, signal);
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

export type AutoLabelStatus = 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';

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
  return apiFetch<AutoLabelJobState>('/curation/pipeline/auto_label/status', {}, signal);
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

// ===========================================================================
// LPR bake-off (/curation/bakeoff) — model comparison runs + results.
// ===========================================================================

export interface BakeoffModelSpec {
  backend: 'ultralytics' | 'triton' | 'open-image-models' | 'two-stage' | 'lpdnet';
  name: string;
  mode?: 'full' | 'crop' | 'both';
  weights?: string;
  imgsz?: number;
  device?: string;
  triton_url?: string;
  triton_model?: string;
  lpdnet_variant?: 'usa' | 'ccpd';
  vehicle_weights?: string;
  training_data?: string;
}

export interface BakeoffRunRow {
  model: string;
  runtime: string;
  training_data: string;
  imgsz: number | string;
  map_50: number;
  map_50_95: number;
  ap_small: number;
  mean_iou: number;
  precision: number;
  recall: number;
  f1: number;
  latency_ms: number;
  fps: number;
}

export interface BakeoffComparison {
  models: BakeoffRunRow[];
  n_models: number;
}

export interface BakeoffRunSummary {
  job_id: string;
  state: string | null;
  models: string[];
  started_at?: string;
  finished_at?: string;
}

/** A finished training run selectable as a bake-off contender. */
export interface BakeoffTrainedModel {
  run_id: string;
  name: string;
  model_size: string | null;
  checkpoint_path: string;
  map50: number | null;
  finished_at: string | null;
  campaign_id: string | null;
}

/**
 * List finished training runs (with a checkpoint) so the bake-off can score an
 * already-trained model straight from the backend — no download/re-upload.
 */
export function bakeoffTrainedModels(
  signal?: AbortSignal,
): Promise<{ models: BakeoffTrainedModel[]; count: number }> {
  return apiFetch('/curation/bakeoff/trained_models', {}, signal);
}

/** A frozen evaluation dataset (a column in the bake-off matrix). */
export interface BakeoffEvalDataset {
  name: string;
  path: string;
  kind: string;
  n_test: number | null;
  frozen_sha: string | null;
}

/** model x dataset matrix from a finished matrix bake-off. */
export interface BakeoffMatrix {
  datasets: string[];
  models: string[];
  metrics: string[];
  cells: Record<string, Record<string, Record<string, number | null>>>;
  best: Record<string, Record<string, string>>;
}

export function bakeoffRun(
  body: {
    dataset?: string;
    datasets?: { path: string; name?: string }[];
    models: BakeoffModelSpec[];
    verify_frozen?: boolean;
    job_id?: string;
  },
  signal?: AbortSignal,
): Promise<{ status: string; job_id: string; out_dir: string }> {
  return apiFetch(
    '/curation/bakeoff/run',
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Auto-discovered frozen evaluation datasets (matrix columns). */
export function bakeoffEvalDatasets(
  signal?: AbortSignal,
): Promise<{ datasets: BakeoffEvalDataset[]; count: number }> {
  return apiFetch('/curation/bakeoff/eval_datasets', {}, signal);
}

/** Public/commercial baseline detectors from the editable registry. */
export function bakeoffBaselineModels(
  signal?: AbortSignal,
): Promise<{ baselines: BakeoffModelSpec[]; count: number }> {
  return apiFetch('/curation/bakeoff/baseline_models', {}, signal);
}

/** The model x dataset matrix for a finished matrix job. */
export function bakeoffMatrix(
  jobId: string,
  signal?: AbortSignal,
): Promise<BakeoffMatrix> {
  return apiFetch(`/curation/bakeoff/matrix/${encodeURIComponent(jobId)}`, {}, signal);
}

export function bakeoffRuns(
  signal?: AbortSignal,
): Promise<{ runs: BakeoffRunSummary[] }> {
  return apiFetch('/curation/bakeoff/runs', {}, signal);
}

export function bakeoffStatus(
  jobId: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  return apiFetch(`/curation/bakeoff/status/${encodeURIComponent(jobId)}`, {}, signal);
}

export function bakeoffResults(
  jobId: string,
  signal?: AbortSignal,
): Promise<BakeoffComparison> {
  return apiFetch(`/curation/bakeoff/results/${encodeURIComponent(jobId)}`, {}, signal);
}
