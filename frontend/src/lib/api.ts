/**
 * Typed API client for the OpenProcessor curation endpoints, mounted
 * under `API_PREFIX` (`/curation`).
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
  parseMethodsResponse,
  type MethodsResponse,
} from './strategies';
import { parseCurationSettings, type CurationSettings } from '$lib/curationSettings';
import { mapCropSlots } from './annotations/cropSlots';
import type { XYXY, SlotKey, SlotData, SlotSpec } from './annotations/types';
import type { DatasetExportSpec } from './annotations/datasetExport';
import type {
  BulkLabelResult,
  ClusterFilter,
  CropFilter,
  RegistryClass,
  RegistryClassCreate,
  RegistryClassMerge,
  RegistryClassUpdate,
  Cluster,
  Crop,
  ExportDatasetList,
  ExportResult,
  ExportStatus,
  ApiHealth,
  SingleClassExportResult,
  SingleClassExportStatus,
  ModelsStatus,
  StatsSummary,
  TestHoldoutFreezeResult,
  TestHoldoutStats,
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
// Empty default: in Docker the labeler's nginx proxies {API_PREFIX}/* to the
// backend on the same docker network. Relative URLs work from any LAN IP / VPN client.
// For local `npm run dev` outside Docker, set PUBLIC_TRITON_API_URL=http://localhost:4603 in .env.
const RAW_BASE = (import.meta.env?.PUBLIC_TRITON_API_URL as string | undefined) ?? '';

export const apiBase: string = RAW_BASE.replace(/\/+$/, '');

/**
 * Path prefix every backend endpoint hangs off, e.g. `/curation/health`.
 * Defaults to OpenProcessor's `OP_API_PREFIX` default, `/curation`, and
 * must equal it: the backend also builds some URLs itself (thumbnail
 * URLs in `/regions` rows) from its own prefix, and nginx only proxies
 * this one. Flipped from a transitional prefix at T-E2
 * (`docs/design/backend-integration-phase-b-plan-2026-09-20.md`).
 */
const RAW_API_PREFIX = (import.meta.env?.PUBLIC_API_PREFIX as string | undefined) ?? '';

/**
 * Empty and `__API_PREFIX__` both mean "unset", on purpose.
 *
 * The production image bakes the literal `__API_PREFIX__` at build time
 * and `sed`s it at container start (`docker-entrypoint.sh`). If the env
 * var is unset there the substitution yields `''`, and `?? '/curation'` would
 * NOT fire — `??` only catches null/undefined — producing prefix-less
 * URLs like `/health`. If the entrypoint is skipped entirely the
 * placeholder leaks through verbatim. Both are total, silent failures;
 * both are cheaper to absorb here than to debug in a container.
 */
export function normalizeApiPrefix(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed || trimmed.startsWith('__')) return '/curation';
  const leading = trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
  return leading.replace(/\/+$/, '');
}

export const API_PREFIX: string = normalizeApiPrefix(RAW_API_PREFIX);

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
    // Structured FastAPI details (`{detail: {error, ...}}`) carry their
    // human-readable text under `error`.
    if (raw && typeof raw === 'object') {
      raw = (raw as Record<string, unknown>).error ?? null;
    }
  }
  if (typeof raw !== 'string') return null;
  const text = raw.trim();
  if (!text) return null;
  return text.length > DETAIL_MAX_CHARS
    ? `${text.slice(0, DETAIL_MAX_CHARS - 1)}…`
    : text;
}

/** The 422 `detail` an unknown per-run strategy id (e.g. `prompt_pack`)
 *  produces on `auto_label/start`. */
export interface UnknownStrategyDetail {
  axis: string;
  requested: string;
  valid_ids: string[];
}

export function unknownStrategyDetail(e: unknown): UnknownStrategyDetail | null {
  if (!(e instanceof ApiError) || e.status !== 422) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (typeof d.axis !== 'string' || typeof d.requested !== 'string') return null;
  if (!Array.isArray(d.valid_ids)) return null;
  return {
    axis: d.axis,
    requested: d.requested,
    valid_ids: d.valid_ids.filter((v): v is string => typeof v === 'string'),
  };
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
 * The backend intentionally emits relative `{API_PREFIX}/...` URLs in API
 * response bodies (e.g. `region_thumbnail_url`, `representative_thumb_urls`)
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

export function getHealth(signal?: AbortSignal): Promise<ApiHealth> {
  return apiFetch<ApiHealth>(`${API_PREFIX}/health`, {}, signal);
}

/**
 * Capability discovery for the curation-strategy registries (plan §3/§5.3):
 * which cluster methods / review sorts / overlays / scores the backend
 * currently offers, each with a `stable | experimental | shadow |
 * disabled` status. Phase 0 plumbing only — nothing consumes this yet.
 *
 * **Never rejects.** `{API_PREFIX}/methods` may not exist yet (backend Phase 0
 * lands independently — see `strategies.ts`'s header), and this endpoint
 * is pure capability discovery, not something a caller should have to
 * try/catch around. `apiFetch` already applies the house retry rule (no
 * retry on 4xx, 3 retries with backoff on 5xx/network errors); once that
 * settles, a 404 or any other failure here resolves to `FALLBACK_METHODS`
 * — the hardcoded stable-only list matching what's actually implemented
 * today — instead of throwing. A caller-initiated abort still propagates,
 * since that's a cancellation, not a backend failure.
 */
export async function getMethods(signal?: AbortSignal): Promise<MethodsResponse> {
  try {
    const raw = await apiFetch<unknown>(`${API_PREFIX}/methods`, {}, signal);
    return parseMethodsResponse(raw);
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    return FALLBACK_METHODS;
  }
}

// -- shared curation defaults (GET,PUT {API_PREFIX}/settings) -----------
//
// Deployment-wide strategy defaults, verified against wt-oss-hardening's
// `src/routers/curation/settings.py` on 2026-09-21. See
// docs/design/curation-settings-ui-plan-2026-09-21.md §1.3.

/**
 * Read the deployment's shared curation defaults.
 *
 * **Unlike `getMethods()`, this DOES reject.** That asymmetry is
 * deliberate: `getMethods` is fired from many component mounts and its
 * absence has a meaningful fallback (`FALLBACK_METHODS`), so swallowing
 * failures there is right. This endpoint is fired from exactly one page,
 * and that page must distinguish three outcomes an opaque fallback would
 * fuse into one:
 *
 *   404  -> this backend predates the feature; show "not supported",
 *           render no controls at all
 *   5xx/net -> transient; show the error and offer a retry
 *   200  -> real record (possibly `defaults: {}` when nothing has ever
 *           been written — that is the normal first-run response, NOT an
 *           error)
 *
 * Throwing preserves `ApiError.status`, which is the only thing that can
 * tell those apart. `curationSettingsStore` is the single place that
 * catches.
 */
export async function getCurationSettings(
  signal?: AbortSignal,
): Promise<CurationSettings> {
  const raw = await apiFetch<unknown>(`${API_PREFIX}/settings`, {}, signal);
  return parseCurationSettings(raw);
}

/**
 * Merge one or more axis defaults into the shared record.
 *
 * PARTIAL BODY BY CONTRACT — send only the axes being changed. The
 * backend writes `{'doc': {...}, 'doc_as_upsert': True}`, an OpenSearch
 * recursive object merge, so axes not mentioned are left untouched. This
 * is also what protects an axis id this build has never heard of from
 * being clobbered by an older frontend: never send the whole `defaults`
 * map back, only the delta.
 *
 * Throws `ApiError` with `status === 422` when an axis is unsettable or
 * an id is not currently advertised for it; `ApiError.detail` carries the
 * server's own message listing the valid axes/ids. Note `errorDetail()`
 * truncates at 200 chars, so a long `valid ids: [...]` list can be
 * elided — the caller should re-sync `/methods` rather than rely on
 * parsing that string (see the store's `saveDefault`).
 *
 * A `null` value for an axis CLEARS that axis's shared override — the
 * backend drops it from the stored record entirely, so the next GET's
 * `defaults` map omits that key and callers fall back to the axis's own
 * built-in default. This is the fix for the gap once tracked as H-1 in
 * docs/design/curation-settings-ui-plan-2026-09-21.md (a pinned default
 * used to be permanent, since PUT only merged and every value had to be
 * a currently-advertised id).
 *
 * Returns the FULL merged record. Always adopt this; never
 * optimistically construct the post-save state client-side.
 */
export async function putCurationDefaults(
  defaults: Record<string, string | null>,
  signal?: AbortSignal,
): Promise<CurationSettings> {
  const raw = await apiFetch<unknown>(
    `${API_PREFIX}/settings`,
    { method: 'PUT', body: JSON.stringify({ defaults }) },
    signal,
  );
  return parseCurationSettings(raw);
}

// -- embedding projection (2-d visualization overlay, Phase 5) -----------
//
// docs/curation-strategy-plan-2026-09.md §2.7/§5.6/§7 — `embedding_viz.py`
// + `viz.py` (openprocessor). UMAP-as-a-visualization-only overlay is the
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
   * CONFIRMED (2026-09-10) against the real `GET {API_PREFIX}/viz/projection`
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
 * `getMethods`'s contract) — `{API_PREFIX}/viz/projection` may not exist yet (the
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
      `${API_PREFIX}/viz/projection${qs({
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
    `${API_PREFIX}/viz/projection/rebuild`,
    { method: 'POST' },
    signal,
  );
}

/** Current rebuild-job snapshot — poll this after `rebuildVizProjection()`. */
export function getVizProjectionStatus(signal?: AbortSignal): Promise<VizProjectionJob> {
  return apiFetch<VizProjectionJob>(`${API_PREFIX}/viz/projection/status`, {}, signal);
}

/** Cancel a running rebuild. `cancelled` is false when nothing was running. */
export function cancelVizProjection(
  signal?: AbortSignal,
): Promise<VizProjectionJob & { cancelled: boolean }> {
  return apiFetch<VizProjectionJob & { cancelled: boolean }>(
    `${API_PREFIX}/viz/projection/cancel`,
    { method: 'POST' },
    signal,
  );
}

// -- plates browse / training-cohort selection ---------------------------

/**
 * Base path for the region/annotation-slot collection endpoints (the
 * license_plate profile's "plates browse" sub-system below — cluster,
 * FP centroids, training candidates, etc). The backend
 * (OpenProcessor/openprocessor) renames this `/plates` -> `/regions`
 * (merged to its `main` at `b3f928d`, 2026-09) — flipping this ONE
 * constant is the whole lockstep change (Wave 2, C13/C14 of
 * docs/design/slot-generic-crop-mapping-plan-2026-09-21.md). Do not
 * reintroduce a bare '/plates' literal in any of the sites below;
 * `regionRouteScan.test.ts` fails the build if you do.
 *
 * Flipped 2026-09-21 (Wave 2 C14) — the backend's rename shipped; this
 * is the whole lockstep change. `git revert` this commit to restore
 * compatibility with a pre-rename backend.
 */
const REGION_BASE = '/regions';

export interface PlateBrowseItem {
  crop_id: string;
  id: string;
  image_path: string;
  bbox_norm: number[];
  region_bbox_norm: number[] | null;
  region_score: number | null;
  region_status: string | null;
  region_verified: boolean | null;
  region_validated: boolean | null;
  region_detector: string | null;
  region_detector_version: string | null;
  region_detector_chain: string[] | null;
  region_bbox_frame: string | null;
  region_detected_at: string | null;
  region_verifier: string | null;
  region_verifier_version: string | null;
  region_verified_at: string | null;
  region_rejection_reason: string | null;
  region_visible: boolean | null;
  region_text: string | null;
  region_text_source: string | null;
  region_text_confidence: number | null;
  class_id: number | null;
  class_name: string | null;
  cluster_id: number | null;
  /** Parent-crop rank by size in its image (1 = largest). */
  crop_rank_in_image?: number | null;
  crop_area_norm?: number | null;
  /** Plate clustering assignment (independent of vehicle cluster_id). */
  region_cluster_id?: number | null;
  region_cluster_subid?: string | null;
  region_cluster_distance?: number | null;
  updated_at: string;
  thumbnail_url?: string;
  region_thumbnail_url?: string;
  selection_reason?: string;
  /** Per-slot capability data — see `Crop.slots` in types.ts. Added by
   *  `getPlates` via `mapCropSlots`; absent on any row that predates this
   *  mapping in a stale cache. */
  slots?: Record<SlotKey, SlotData>;
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
  region_cluster_id?: number;
  /** AHC plate sub-cluster id (e.g. "17a"). */
  region_cluster_subid?: string;
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

/** `browsePath` is the slot's declared browse collection
 *  (`capabilities.queue.browsePath`), so a slot never inherits another
 *  slot's route by accident. */
export async function getPlates(
  browsePath: string,
  params: PlatesQuery = {},
  signal?: AbortSignal,
): Promise<PlatesPage> {
  const page = await apiFetch<PlatesPage>(
    `${API_PREFIX}${browsePath}${qs(params as Record<string, unknown>)}`,
    {},
    signal,
  );
  return {
    ...page,
    items: page.items.map((raw) => ({
      ...raw,
      slots: mapCropSlots(
        raw as unknown as Record<string, unknown>,
        (raw.bbox_norm ?? [0, 0, 0, 0]) as XYXY,
      ),
    })),
  };
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
        n_regions?: number;
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
    `${API_PREFIX}${REGION_BASE}/cluster${qs({
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
  return apiFetch(`${API_PREFIX}${REGION_BASE}/cluster/status`, {}, signal);
}

/** Per-bucket AHC refine over plate_pe_embedding; writes region_cluster_subid. */
export function refinePlateCluster(
  clusterId: number,
  signal?: AbortSignal,
): Promise<{
  cluster_id: number;
  n_members: number;
  n_subclusters: number;
  action: string;
}> {
  return apiFetch(
    `${API_PREFIX}${REGION_BASE}/clusters/refine/${clusterId}`,
    { method: 'POST' },
    signal,
  );
}

/** Plate cluster cards (mirrors getClusters' Cluster shape). */
export function getPlateClusters(
  opts: { maxClusters?: number; perCluster?: number; maxRank?: number } = {},
  signal?: AbortSignal,
): Promise<{ clusters: Cluster[]; count: number }> {
  return apiFetch(
    `${API_PREFIX}${REGION_BASE}/clusters${qs({
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
  return apiFetch(
    `${API_PREFIX}${REGION_BASE}/fp_centroids/build`,
    { method: 'POST' },
    signal,
  );
}

/** Poll the FP-centroid build job + read persisted centroid metadata. */
export function getPlateFpCentroidStatus(
  signal?: AbortSignal,
): Promise<PlateFpCentroidJob> {
  return apiFetch(`${API_PREFIX}${REGION_BASE}/fp_centroids/status`, {}, signal);
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
    `${API_PREFIX}${REGION_BASE}/suspected_false_positives${qs({
      threshold: opts.threshold,
      page: opts.page,
      page_size: opts.pageSize,
    })}`,
    {},
    signal,
  );
}

// Wave 2 C14: 'lpr_blind_spots' -> 'detector_blind_spots',
// 'lpr_low_conf_correct' -> 'low_conf_correct' (the wire ?mode= values
// licensePlateSlot's cohorts now send — see licensePlate.ts).
export type TrainingCohortMode =
  | 'detector_blind_spots'
  | 'low_conf_correct'
  | 'disagreement'
  | 'human_corrected'
  | 'false_positives';

export function getTrainingCandidates(
  mode: TrainingCohortMode,
  params: { page?: number; page_size?: number; class_id?: number } = {},
  signal?: AbortSignal,
): Promise<PlatesPage> {
  return apiFetch<PlatesPage>(
    `${API_PREFIX}${REGION_BASE}/training_candidates${qs({ mode, ...params })}`,
    {},
    signal,
  );
}

export function getModelsStatus(signal?: AbortSignal): Promise<ModelsStatus> {
  return apiFetch<ModelsStatus>(`${API_PREFIX}/models/status`, {}, signal);
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
    `${API_PREFIX}/models/${encodeURIComponent(modelName)}${qs({ force })}`,
    { method: 'DELETE' },
    signal,
  );
}

/**
 * Pipeline-dashboard payload from `GET {API_PREFIX}/stats/dataset`. Contract
 * defined by `src/routers/curation/stats.py` — every nested key is
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
    by_vlm: number;
    by_classifier: number;
    by_proposal: number;
    other: number;
  };
  regions: {
    /** Crops with a region_bbox_norm right now — the honest "crops with a
     *  plate" count (matches the plate cluster view). */
    boxed?: number;
    /** Crops Gemma confirmed are real plates (region_status='detected'). */
    confirmed?: number;
    /** Sum of region_detector credit — includes rejected/failed attempts,
     *  so it OVERSTATES real plates. Kept for back-compat; not the headline. */
    total_detected: number;
    by_detector: number;
    by_segmenter: number;
    /** Legacy alias for ``by_human_drew``. */
    by_human: number;
    /** Crops where the operator drew a fresh plate bbox from scratch. */
    by_human_drew?: number;
    /** Crops whose plate was verified by a human (Confirm Plate button). */
    verified_by_human?: number;
    /** Crops whose plate was verified by Gemma (auto-verify). */
    verified_by_vlm?: number;
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
  return apiFetch<DatasetStats>(`${API_PREFIX}/stats/dataset`, {}, signal);
}

export async function getStats(signal?: AbortSignal): Promise<StatsSummary> {
  // The API returns
  //   {API_PREFIX}/stats/dataset:  {total_crops, validated, test_holdout, by_source}
  //   {API_PREFIX}/stats/classes:  {classes:[{class_id, class_name, count, validated_count}, ...]}
  // The labeler dashboard expects StatsSummary which uses validated_crops /
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
    apiFetch<RawDataset>(`${API_PREFIX}/stats/dataset`, {}, signal),
    apiFetch<RawClasses>(`${API_PREFIX}/stats/classes`, {}, signal).catch(() => ({
      classes: [],
    })),
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

export async function getClasses(signal?: AbortSignal): Promise<RegistryClass[]> {
  // The API returns `{classes: [{class_id, class_name, group, sample_count,
  // validated_count, deprecated}, ...]}`. Map to the labeler's RegistryClass
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
    `${API_PREFIX}/classes`,
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

/** Raw cluster card from `{API_PREFIX}/clusters`. The backend is the single
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

function _rawClusterToCluster(c: RawCluster): Cluster {
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
): Promise<PaginatedResponse<Cluster>> {
  // Single round-trip. The backend's {API_PREFIX}/clusters aggregation already
  // returns dominant class, purity, validated_count, n_subclusters,
  // cluster_kind, and is_unlabeled. The frontend ONLY shapes the result
  // into the labeler's Cluster type — no semantic compute here.
  const raw = await apiFetch<RawClustersResp>(
    `${API_PREFIX}/clusters${qs({
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
  const items = (raw.items ?? []).map(_rawClusterToCluster);
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

/** Raw crop shape from the {API_PREFIX}/crops API. */
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
  class_detector?: string | null;
  class_detector_version?: string | null;
  class_labeled_at?: string | null;
  class_labeler?: string | null;
  test_holdout?: boolean;
  crop_rank_in_image?: number | null;
  crop_area_norm?: number | null;
  blur_lap_ratio?: number | null;
  classifier_raw_confidence?: number | null;
  proposal_name?: string | null;
  vlm_confidence?: string | null;
  vlm_proposed_class_id?: number | null;
  vlm_proposed_class_name?: string | null;
  // Curation scores (Phase 3, docs/curation-strategy-plan-2026-09.md §4).
  // Optional/forward-tolerant: an un-backfilled pool just omits these.
  mistakenness_score?: number | null;
  mistakenness_method?: string | null;
  mistakenness_version?: string | null;
  mistakenness_scored_at?: string | null;
  thumbnail_url?: string;
  updated_at?: string;
};

function mapRawCrop(c: RawCrop): Crop {
  const bb = c.bbox_norm ?? [0, 0, 0, 0];
  const out: Crop = {
    id: c.crop_id,
    source_image_path: c.image_path,
    bbox_norm: xyxyToBBoxNorm(bb),
    class_id: c.class_id ?? null,
    class_name: c.class_name ?? null,
    class_source: c.class_source ?? null,
    label_source: c.label_source || 'unknown',
    label_validated: !!c.label_validated,
    label_confidence: c.confidence ?? null,
    cluster_id: c.cluster_id ?? null,
    similarity_to_centroid:
      c.cluster_distance != null ? Math.max(0, 1 - c.cluster_distance) : null,
    cluster_subid: c.cluster_subid ?? null,
    class_detector: c.class_detector ?? null,
    class_detector_version: c.class_detector_version ?? null,
    class_labeled_at: c.class_labeled_at ?? null,
    class_labeler: c.class_labeler ?? null,
    test_holdout: !!c.test_holdout,
    crop_rank_in_image: c.crop_rank_in_image ?? null,
    crop_area_norm: c.crop_area_norm ?? null,
    blur_lap_ratio: c.blur_lap_ratio ?? null,
    classifier_raw_confidence: c.classifier_raw_confidence ?? null,
    proposal_name: c.proposal_name ?? null,
    vlm_confidence: c.vlm_confidence ?? null,
    vlm_suggested_class_id: c.vlm_proposed_class_id ?? null,
    vlm_suggested_class_name: c.vlm_proposed_class_name ?? null,
    mistakenness_score: c.mistakenness_score ?? null,
    mistakenness_method: c.mistakenness_method ?? null,
    mistakenness_version: c.mistakenness_version ?? null,
    mistakenness_scored_at: c.mistakenness_scored_at ?? null,
    slots: mapCropSlots(c as unknown as Record<string, unknown>, bb as XYXY),
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
    classifierConfLt?: number | null;
    /**
     * Forwarded verbatim to `{API_PREFIX}/crops?order=`. Only `'outliers'` is
     * special-cased server-side today (crops.py `order` query param —
     * see docs/curation-strategy-plan-2026-09.md §1); an id the backend
     * doesn't recognize is harmless (qs() still sends it, the server
     * just falls back to its default ordering). Typed as `string` rather
     * than a fixed union so a new `{API_PREFIX}/methods`-reported order id doesn't
     * require touching this signature — callers should still gate which
     * ids they actually offer against what `{API_PREFIX}/methods` reports.
     */
    order?: string | null;
    /**
     * Pool-scale overlay parameter, forwarded to `{API_PREFIX}/crops?k=` only when
     * set (curation-strategy plan Phase 4 — `order: 'diverse'`'s "how
     * many diverse crops" count). Meaningless for every other `order`
     * value; the caller (`/clusters/[id]`) only sets it in diverse mode.
     */
    k?: number | null;
  } = {},
): Promise<{ cluster: Cluster; crops: PaginatedResponse<Crop> }> {
  // Two parallel calls: paginated crops + the authoritative cluster
  // card from {API_PREFIX}/clusters (server-computed). The page no longer
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
  if (opts.classifierConfLt != null) cropQuery.classifier_conf_lt = opts.classifierConfLt;
  if (opts.order) cropQuery.order = opts.order;
  if (opts.k != null) cropQuery.k = opts.k;
  const [cropPage, clustersResp] = await Promise.all([
    apiFetch<CropPage>(`${API_PREFIX}/crops${qs(cropQuery)}`, {}, signal),
    apiFetch<RawClustersResp>(
      // cluster_id (not class_id!) is the correct filter for "fetch this
      // one cluster's card by its own identity" — cluster_id == class_id
      // is only an invariant for 'class' clusters, so filtering by
      // class_id silently returned nothing for every 'candidate' cluster
      // (id >= RESIDUAL_CLUSTER_ID_OFFSET, no matching class exists),
      // which fell back to the null-identity stub below and showed no
      // human-readable name in the header even though {API_PREFIX}/clusters'
      // list view has dominant_class_name for the same cluster.
      `${API_PREFIX}/clusters${qs({ per_cluster: 4, max_clusters: 1, cluster_id: id })}`,
      {},
      signal,
    ).catch(() => null),
  ]);
  const items = cropPage.crops.map(mapRawCrop);
  const found = clustersResp?.items?.find((c) => c.cluster_id === id) ?? null;
  const cluster: Cluster = found
    ? _rawClusterToCluster(found)
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
): Promise<PaginatedResponse<Crop>> {
  type Raw = { total: number; page: number; page_size: number; crops: RawCrop[] };
  const raw = await apiFetch<Raw>(`${API_PREFIX}/crops${qs({ ...filter })}`, {}, signal);
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
): Promise<Crop> {
  return apiFetch<Crop>(
    `${API_PREFIX}/crops/${encodeURIComponent(cropId)}/label`,
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
    `${API_PREFIX}/crops/batch_label`,
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
    `${API_PREFIX}/crops/${encodeURIComponent(cropId)}/label`,
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
    `${API_PREFIX}/crops/${encodeURIComponent(cropId)}/review_dismiss`,
    { method: 'POST' },
    signal,
  );
}

/**
 * Update or clear the plate sub-bbox on a crop.
 *
 * - Pass an `[x1, y1, x2, y2]` tuple in **source-image normalized**
 *   coordinates to set/replace the plate box (server records
 *   `region_status='human_confirmed'`).
 * - Pass `null` to clear the plate; the backend interprets this as
 *   `region_status='no_region_visible'`.
 *
 * Mirrors `putCropLabel` in shape. Endpoint: `PUT {API_PREFIX}/crops/{id}/region`
 * (renamed from `/plate`, Wave 2 C14), defined by backend task #32 to
 * match this contract.
 *
 * Dead code as of Wave 1 (C6/C8): every call site now goes through
 * `setSlotBox(licensePlateSlot, …)` instead. Kept renamed rather than
 * deleted here — deleting it is orthogonal to the wire rename.
 */
/**
 * Fetch a single crop by id from the authoritative store. Used by the
 * review-page "Back" path so the operator sees what was actually
 * persisted rather than a possibly-stale local snapshot. Endpoint:
 * `GET {API_PREFIX}/crops/{crop_id}`.
 */
export async function getCrop(cropId: string, signal?: AbortSignal): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${API_PREFIX}/crops/${encodeURIComponent(cropId)}`,
    {},
    signal,
  );
  return mapRawCrop(raw);
}

/**
 * PUT a slot's sub-box via the spec's declared endpoint, or clear it
 * (`xyxy === null`) via `clearBox` when the profile declares a distinct
 * one, falling back to `setBox` with a null box otherwise (the backend's
 * "PUT with a null box clears" contract). The body key is the slot's own
 * `subBox.bboxField`, so a slot's writes use the same wire name its reads
 * do.
 */
export function setSlotBox(
  spec: SlotSpec,
  cropId: string,
  xyxy: [number, number, number, number] | null,
  signal?: AbortSignal,
): Promise<Crop> {
  const path =
    (xyxy === null ? spec.endpoints.clearBox?.(cropId) : undefined) ??
    spec.endpoints.setBox?.(cropId);
  const bboxField = spec.capabilities.subBox?.bboxField;
  if (!path || !bboxField) {
    return Promise.reject(
      new Error(`slot "${spec.key}" has no setBox/clearBox endpoint or subBox field`),
    );
  }
  return apiFetch<Crop>(
    `${API_PREFIX}${path}`,
    { method: 'PUT', body: JSON.stringify({ [bboxField]: xyxy }) },
    signal,
  );
}

/**
 * PATCH a slot's metadata fields (status / text / rejection reason)
 * without touching the bbox. The BODY KEYS are the spec's own wire
 * field names — for `licensePlateSlot` this produces a body
 * byte-identical to the old `PlateMetaPatch` (`region_text` /
 * `region_status` / `region_rejection_reason`), pinned in api.test.ts.
 * Keys whose capability is absent, or whose value is `undefined`
 * (as opposed to `null`, which clears), are omitted.
 */
export function patchSlotMeta(
  spec: SlotSpec,
  cropId: string,
  patch: {
    status?: string | null;
    text?: string | null;
    rejectionReason?: string | null;
  },
  signal?: AbortSignal,
): Promise<{ crop_id: string; updated_fields: string[] }> {
  const cap = spec.capabilities;
  const body: Record<string, unknown> = {};
  if (patch.status !== undefined && cap.lifecycle?.statusField) {
    body[cap.lifecycle.statusField] = patch.status;
  }
  if (patch.text !== undefined && cap.text?.valueField) {
    body[cap.text.valueField] = patch.text;
  }
  if (patch.rejectionReason !== undefined && cap.lifecycle?.rejectionReasonField) {
    body[cap.lifecycle.rejectionReasonField] = patch.rejectionReason;
  }
  const path = spec.endpoints.patchMeta?.(cropId);
  if (!path) {
    return Promise.reject(new Error(`slot "${spec.key}" has no patchMeta endpoint`));
  }
  return apiFetch(
    `${API_PREFIX}${path}`,
    { method: 'PATCH', body: JSON.stringify(body) },
    signal,
  );
}

/**
 * Bulk-set region_status over many crops. Backend: `POST {API_PREFIX}/regions/batch_status`
 *   (renamed from `{API_PREFIX}/plates/batch_status`, Wave 2 C14).
 * The cluster-view triage op: select outlier plates → mark all false_positive,
 * or bulk-confirm good plates (status='detected' + plateVerified=true).
 *
 * Reads its path from `spec.endpoints.batchStatus` (Wave 2 C13,
 * docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §8.2 trap 2)
 * rather than hardcoding `${REGION_BASE}/batch_status` a second time —
 * closes the "declared but dead" gap without importing a specific
 * profile into this generic module (falls back to the REGION_BASE path
 * if a spec declares no batchStatus endpoint, matching today's only caller).
 */
export function batchPlateStatus(
  spec: SlotSpec,
  cropIds: string[],
  plateStatus: 'detected' | 'no_region_visible' | 'verify_rejected' | 'false_positive',
  opts: { plateVerified?: boolean; labelSource?: string } = {},
  signal?: AbortSignal,
): Promise<{
  updated: number;
  conflicts: { crop_id: string; current_source: string | null }[];
}> {
  const path = spec.endpoints.batchStatus?.() ?? `${REGION_BASE}/batch_status`;
  const lc = spec.capabilities.lifecycle;
  if (!lc) {
    return Promise.reject(new Error(`slot "${spec.key}" has no lifecycle capability`));
  }
  const body: Record<string, unknown> = {
    crop_ids: cropIds,
    [lc.statusField]: plateStatus,
  };
  if (lc.verifiedField) body[lc.verifiedField] = opts.plateVerified ?? null;
  if (lc.labelSourceField) body[lc.labelSourceField] = opts.labelSource ?? 'human';
  return apiFetch(
    `${API_PREFIX}${path}`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

export async function runVlmOnCluster(
  clusterId: number,
  signal?: AbortSignal,
): Promise<{ predicted: number; updated: number; new_class_proposals?: unknown[] }> {
  // {API_PREFIX}/vlm/label_batch takes {crop_ids: [...]} (max 64) — the
  // backend renamed the path segment gemma → vlm when it swapped Gemma
  // for a pluggable VLM abstraction. The JSON field names (gemma_*) and
  // the vlm_low_conf review-tab id are frozen wire contract and did
  // NOT move. Fetch the
  // unvalidated crops in this cluster first, then POST in chunks of 64.
  type CropPage = { crops: Array<{ crop_id: string }> };
  const page = await apiFetch<CropPage>(
    `${API_PREFIX}/crops${qs({ cluster_id: clusterId, label_validated: false, page_size: 200 })}`,
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
      `${API_PREFIX}/vlm/label_batch`,
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
    `${API_PREFIX}/clusters/refine/${clusterId}`,
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
  // The {API_PREFIX}/review API ships bbox_norm + region_bbox_norm as
  // [x1,y1,x2,y2] arrays. The labeler's ReviewItem extends Crop where
  // bboxes are {cx,cy,w,h} objects. Normalize each item through
  // mapRawCrop so SlotBboxEditor + getThumbUrl + confirmSlot all see the
  // same shape regardless of the endpoint that produced the item.
  type RawReviewItem = RawCrop & {
    reason?: string;
    proposed_class_id?: number | null;
    proposed_class_name?: string | null;
    probe_pred_class?: string | null;
    probe_pred_entropy?: number | null;
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
    `${API_PREFIX}/review/${tab}${qs({ page, page_size: pageSize, ...filter })}`,
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
// `POST {API_PREFIX}/select/diverse` is a DIFFERENT contract from `/clusters/[id]`'s
// `GET {API_PREFIX}/crops?order=diverse&k=N`: that path is a small, synchronous,
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
      `${API_PREFIX}/select/diverse`,
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
    return {
      kind: 'ready',
      selection: { crop_ids: [], method: 'diverse', version: '', n_pool: 0 },
    };
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
  const raw = await apiFetch<unknown>(`${API_PREFIX}/select/status`, {}, signal);
  return parseSelectJobStatus(raw);
}

/** Cancel the singleton diverse-selection job, if any is running. Real
 *  backend returns `{cancelled: bool, ...job state}` (select.py's
 *  `select_cancel`), not a bare 204 — the caller only needs to know
 *  polling can stop, so the body is discarded. */
export async function cancelSelect(signal?: AbortSignal): Promise<void> {
  await apiFetch<unknown>(`${API_PREFIX}/select/cancel`, { method: 'POST' }, signal);
}

/**
 * Free-text semantic search over vehicle crops (P2-14). Backend:
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
  type RawSearchItem = RawCrop & {
    similarity_score?: number | null;
    semantic_score?: number | null;
    score?: number | null;
  };
  type RawPage = {
    total: number;
    page: number;
    page_size: number;
    items: RawSearchItem[];
  };
  const raw = await apiFetch<RawPage>(
    `${API_PREFIX}/search/text${qs({ q, page, page_size: pageSize, ...filter })}`,
    {},
    signal,
  );
  const items: SearchCrop[] = (raw.items ?? []).map((it) => {
    const base = mapRawCrop(it);
    return {
      ...base,
      // The backend's `_hydrate_item` (openprocessor semantic_search.py) sends
      // the match score as `semantic_score` — `similarity_score`/`score`
      // are legacy/defensive fallbacks that the live endpoint has never
      // actually populated. Without the semantic_score read here every
      // search-result badge silently rendered 0%.
      similarity_score: it.similarity_score ?? it.semantic_score ?? it.score ?? 0,
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
): Promise<ExportResult> {
  const body: Record<string, unknown> = {};
  if (opts.version_tag) body.version_tag = opts.version_tag;
  return apiFetch<ExportResult>(
    `${API_PREFIX}/export/yolo`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Poll current export state. */
export function exportStatus(signal?: AbortSignal): Promise<ExportStatus> {
  return apiFetch<ExportStatus>(`${API_PREFIX}/export/status`, {}, signal);
}

/** Options an operator picks per single-class export build. */
export interface SingleClassExportOptions {
  version_tag?: string;
  empty_bg_ratio?: number;
  max_positive_images?: number;
  skip_test_split?: boolean;
  dedup_threshold?: number | null;
  image_mode?: 'whole_frame' | 'item_crop';
  img_max_side?: 640 | 1280;
}

/**
 * Build a narrowed single-class dataset from a slot's declared export
 * (`DatasetExportSpec`). Synchronous on the server; returns the export
 * dir + counts when done. Only ever called behind
 * `isDatasetExportAvailable(…, spec.kind)` — see `/train`'s gate.
 */
export function exportSingleClass(
  spec: DatasetExportSpec,
  opts: SingleClassExportOptions = {},
  signal?: AbortSignal,
): Promise<SingleClassExportResult> {
  const body: Record<string, unknown> = {
    profile_name: spec.profileName,
    box_source: spec.boxSource,
    class_ids: spec.classIds,
  };
  if (spec.regionClassName) body.region_class_name = spec.regionClassName;
  for (const [k, v] of Object.entries(opts)) {
    if (v !== undefined) body[k] = v;
  }
  return apiFetch<SingleClassExportResult>(
    `${API_PREFIX}${spec.buildPath}`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Last build of a slot's single-class export (its own `current` symlink). */
export function exportSingleClassStatus(
  spec: DatasetExportSpec,
  signal?: AbortSignal,
): Promise<SingleClassExportStatus> {
  return apiFetch<SingleClassExportStatus>(
    `${API_PREFIX}${spec.statusPath}${qs({ profile_name: spec.profileName })}`,
    {},
    signal,
  );
}

/**
 * List every materialized dataset version on disk (newest first) so the
 * operator can train on any past export — a small sample, a larger subset, or
 * the full set — for consistent data re-use across model-size upgrades.
 */
export function listDatasets(
  filter: { kind?: string; profile_name?: string } = {},
  signal?: AbortSignal,
): Promise<ExportDatasetList> {
  return apiFetch<ExportDatasetList>(
    `${API_PREFIX}/export/datasets${qs(filter)}`,
    {},
    signal,
  );
}

// -- classes mutators ----------------------------------------------------

export function getClass(classId: number, signal?: AbortSignal): Promise<RegistryClass> {
  return apiFetch<RegistryClass>(`${API_PREFIX}/classes/${classId}`, {}, signal);
}

export function addClass(
  payload: RegistryClassCreate,
  signal?: AbortSignal,
): Promise<{ class_id: number; class_name: string; group: string }> {
  return apiFetch<{ class_id: number; class_name: string; group: string }>(
    `${API_PREFIX}/classes`,
    { method: 'POST', body: JSON.stringify(payload) },
    signal,
  );
}

export function renameClass(
  classId: number,
  payload: RegistryClassUpdate,
  signal?: AbortSignal,
): Promise<unknown> {
  return apiFetch<unknown>(
    `${API_PREFIX}/classes/${classId}`,
    { method: 'PUT', body: JSON.stringify(payload) },
    signal,
  );
}

export function mergeClasses(
  payload: RegistryClassMerge,
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
  }>(
    `${API_PREFIX}/classes/merge`,
    { method: 'POST', body: JSON.stringify(payload) },
    signal,
  );
}

/** What kind of writer a `class_source` value names. */
export type ClassSourceRole =
  | 'proposal'
  | 'low_conf'
  | 'model'
  | 'vlm'
  | 'vlm_unmatched'
  | 'vlm_new_class_pending'
  | 'vlm_reclassified'
  | 'cluster'
  | 'human'
  | 'merge'
  | 'label_import'
  | (string & {});

/** One `class_source` value this deployment can write. */
export interface ClassSource {
  id: string;
  label: string;
  role: ClassSourceRole;
}

/** `GET {API_PREFIX}/class_sources` — every `class_source` value this
 *  deployment can write, including the ingest detectors' config-derived
 *  ones (`{primary}_proposal`, …). */
export async function getClassSources(signal?: AbortSignal): Promise<ClassSource[]> {
  const res = await apiFetch<{ class_sources?: ClassSource[] }>(
    `${API_PREFIX}/class_sources`,
    {},
    signal,
  );
  return (res.class_sources ?? []).filter(
    (c) => typeof c?.id === 'string' && c.id.length > 0 && typeof c.label === 'string',
  );
}

export function syncClassesToOpensearch(
  signal?: AbortSignal,
): Promise<{ upserted: number; n_classes: number }> {
  return apiFetch<{ upserted: number; n_classes: number }>(
    `${API_PREFIX}/classes/sync_to_opensearch`,
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
  // {API_PREFIX}/crops/batch_label. Reuse the type so both call sites share the
  // conflict-handling code path.
  return apiFetch<BulkLabelResult>(
    `${API_PREFIX}/crops/move`,
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
    `${API_PREFIX}/crops/batch_exclude`,
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
    `${API_PREFIX}/crops/batch_unexclude`,
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
    `${API_PREFIX}/crops/flag_new_class`,
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
): Promise<TestHoldoutFreezeResult> {
  return apiFetch<TestHoldoutFreezeResult>(
    `${API_PREFIX}/test_holdout/freeze`,
    { method: 'POST', body: JSON.stringify(payload) },
    signal,
  );
}

export function getTestHoldoutStats(signal?: AbortSignal): Promise<TestHoldoutStats> {
  return apiFetch<TestHoldoutStats>(`${API_PREFIX}/test_holdout/stats`, {}, signal);
}

// -- registry/manifest downloads (used as anchor `download` URLs) --------

export function getClassRegistryUrl(): string {
  return `${apiBase}${API_PREFIX}/export/registry/class_registry.json`;
}

export function getDataYamlUrl(): string {
  return `${apiBase}${API_PREFIX}/export/registry/data.yaml`;
}

export function getManifestUrl(): string {
  return `${apiBase}${API_PREFIX}/export/registry/manifest.json`;
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
  return `${apiBase}${API_PREFIX}/crops/${encodeURIComponent(cropId)}/thumbnail?size=${size}`;
}

/**
 * URL for an annotation-slot sub-bbox close-up (the region rendered to a
 * tile), same construction convention as {@link getThumbUrl}. Pass
 * `cacheBustKey` (e.g. `Date.now()`) after a bbox edit so the browser
 * doesn't serve the pre-edit crop from its image cache.
 *
 * The segment is `region_thumbnail`, which is the ONLY region-thumbnail
 * route the backend registers (`curation_images.py`'s
 * `@crops_router.get('/{crop_id}/region_thumbnail')`). It used to be
 * `plate_thumbnail`, which 404s — there is no alias and there will not
 * be one (cropwright_backend_integration_plan.md §3.2: no compatibility
 * surface lands on the contract-owning side).
 *
 * Note the deliberate asymmetry with the JSON key: `/regions` responses
 *   (renamed from `/plates`, Wave 2 C14)
 * carry a field literally named `region_thumbnail_url` whose *value* now
 * points at `…/region_thumbnail`. The key is frozen wire contract; only
 * the path inside it is generic. Do not "fix" the key to match.
 */
export function getRegionThumbUrl(
  cropId: string,
  size: number = 160,
  cacheBustKey?: string | number | null,
): string {
  const base = `${apiBase}${API_PREFIX}/crops/${encodeURIComponent(cropId)}/region_thumbnail?size=${size}`;
  return cacheBustKey != null ? `${base}&v=${encodeURIComponent(cacheBustKey)}` : base;
}

export function getSourceImageUrl(cropId: string): string {
  return `${apiBase}${API_PREFIX}/crops/${encodeURIComponent(cropId)}/image`;
}

/**
 * Source image with bbox overlay, downscaled to ~1280px on the longest
 * side. The review page only needs the bbox to be readable, not pixel-
 * perfect — full resolution would push 2+ MB per cursor change. Callers
 * that need a pixel-accurate frame (e.g. SlotBboxEditor) should hit
 * ``getSourceImageFull`` so the bbox lines up with the editor canvas.
 */
export function getSourceImageWithBbox(
  cropId: string,
  maxDim: number = 1280,
  cacheKey?: string | null,
): string {
  // The server reads region_bbox_norm from OpenSearch and burns the
  // overlay into the JPEG. The crop_id alone produces an identical URL
  // across edits, so the browser cache returns the pre-edit JPEG and
  // the left-side preview lags the right-side canvas. Pass a key that
  // changes when the bbox changes (e.g. the bbox tuple) to bust the
  // cache on edits while still hitting the cache between cursor moves.
  const base = `${apiBase}${API_PREFIX}/crops/${encodeURIComponent(cropId)}/image?max_dim=${maxDim}`;
  return cacheKey ? `${base}&v=${encodeURIComponent(cacheKey)}` : base;
}

/** Full-resolution source image; used by SlotBboxEditor where pixel accuracy matters. */
export function getSourceImageFull(cropId: string): string {
  return `${apiBase}${API_PREFIX}/crops/${encodeURIComponent(cropId)}/image`;
}

// -- training endpoints --------------------------------------------------
//
// Mirror the FastAPI `{API_PREFIX}/train/*` router. The labeler `/train` page is
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
    `${API_PREFIX}/train/preflight`,
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
    `${API_PREFIX}/train/start${qs({ force: force ? true : undefined })}`,
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
    `${API_PREFIX}/train/start_campaign${qs({ force: force ? true : undefined })}`,
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
    ? `${API_PREFIX}/train/status/${encodeURIComponent(jobId)}`
    : `${API_PREFIX}/train/status`;
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
    `${API_PREFIX}/train/runs${qs({ limit, offset })}`,
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
    `${API_PREFIX}/train/log/tail/${encodeURIComponent(jobId)}${qs({ lines })}`,
    {},
    signal,
  );
}

export function cancelTrainJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<CancelResponse> {
  return apiFetch<CancelResponse>(
    `${API_PREFIX}/train/cancel/${encodeURIComponent(jobId)}`,
    { method: 'POST' },
    signal,
  );
}

export function cancelTrainCampaign(
  campaignId: string,
  signal?: AbortSignal,
): Promise<CancelResponse> {
  return apiFetch<CancelResponse>(
    `${API_PREFIX}/train/cancel_campaign/${encodeURIComponent(campaignId)}`,
    { method: 'POST' },
    signal,
  );
}

export function getTrainProfiles(signal?: AbortSignal): Promise<ProfilesResponse> {
  return apiFetch<ProfilesResponse>(`${API_PREFIX}/train/profiles`, {}, signal);
}

export function getTrainPresets(signal?: AbortSignal): Promise<PresetsResponse> {
  return apiFetch<PresetsResponse>(`${API_PREFIX}/train/presets`, {}, signal);
}

export function promoteTrainJob(
  jobId: string,
  body: PromoteRequest,
  signal?: AbortSignal,
): Promise<PromoteResponse> {
  return apiFetch<PromoteResponse>(
    `${API_PREFIX}/train/promote/${encodeURIComponent(jobId)}`,
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
    `${API_PREFIX}/train/manifest/${encodeURIComponent(jobId)}`,
    {},
    signal,
  );
}

// -- Auto-label (recluster) job ------------------------------------------
// Wraps POST {API_PREFIX}/pipeline/auto_label/{start,status,cancel}. The pipeline
// re-runs prototype assignment → cluster_id normalize → AHC residuals → auto-
// promote → Gemma sweep, fixing prototype drift and stale cluster_id on
// labeled crops. Hours at HDD scale; the panel polls status while it runs.

export interface AutoLabelStartParams {
  train_clusters?: boolean;
  promote_min_purity?: number;
  promote_min_members?: number;
  vlm_batch_size?: number;
  vlm_concurrency?: number;
  max_vlm_crops?: number;
  classifier_confidence_skip_vlm?: number;
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
  // -- VLM-assisted scoping (2026-09-20 contract, this plan §1.3) --------
  /**
   * Scope the run to a single class instead of the whole pool — "just
   * help me with forklifts right now". Same `class_id` filter convention
   * as `getCrops`/`getClusters`/`/select/diverse` elsewhere in this
   * file. `null`/omitted = today's unscoped, whole-dataset behavior, and
   * `qs()` drops it entirely so an unscoped request stays byte-identical
   * to every request this app has ever sent.
   *
   * Accepted by OpenProcessor `main`'s `auto_label/start`. The UI never
   * sends it unless `/methods` advertises the assist axes — see
   * `isScopedAssistAvailable` in `$lib/strategies`.
   */
  class_id?: number | null;
  /**
   * Per-run override of the settings-doc prompt-pack default, for this
   * job only — omitted/null means the deployment default. An unknown id
   * is a 422 (`unknownStrategyDetail`); the resolved value is echoed in
   * the job state's `args`. Produced in exactly one place
   * (`createAssistScope().toStartParams()` in `$lib/assistScope.svelte`).
   */
  prompt_pack?: string | null;
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
    `${API_PREFIX}/pipeline/auto_label/start${qs(params as Record<string, unknown>)}`,
    { method: 'POST' },
    signal,
  );
}

export function getAutoLabelStatus(signal?: AbortSignal): Promise<AutoLabelJobState> {
  return apiFetch<AutoLabelJobState>(
    `${API_PREFIX}/pipeline/auto_label/status`,
    {},
    signal,
  );
}

export function cancelAutoLabel(
  signal?: AbortSignal,
): Promise<AutoLabelJobState & { cancelled: boolean }> {
  return apiFetch<AutoLabelJobState & { cancelled: boolean }>(
    `${API_PREFIX}/pipeline/auto_label/cancel`,
    { method: 'POST' },
    signal,
  );
}

// ===========================================================================
// Detector bake-off ({API_PREFIX}/bakeoff) — model comparison runs + results.
// ===========================================================================

export interface BakeoffModelSpec {
  backend:
    | 'ultralytics'
    | 'triton'
    | 'open-image-models'
    | 'two-stage'
    | 'lpdnet'
    | 'onnxruntime'
    | 'coreml';
  name: string;
  /** Per-model BakeoffProfile override; else the request-level `profile`. */
  profile?: string;
  mode?: 'full' | 'crop' | 'both';
  weights?: string;
  imgsz?: number;
  device?: string;
  pred_class_id?: number;
  gt_class_id?: number;
  gt_class_name?: string;
  triton_url?: string;
  triton_model?: string;
  lpdnet_variant?: 'usa' | 'ccpd';
  /** Coarse (parent-object) stage for crop / two-stage; unset fields come
   *  from the profile's `context_*`. */
  primary_weights?: string;
  primary_classes?: string;
  primary_imgsz?: number;
  secondary_backend?: string;
  secondary_imgsz?: number;
  training_data?: string;
}

/**
 * What a bake-off scores: the target class under test plus the cascade
 * context and metric config (backend `BakeoffProfile`). `registered`
 * profiles are deployment-configured; `example` ones ship with the
 * harness as templates.
 */
export interface BakeoffProfile {
  name: string;
  kind: 'registered' | 'example';
  target_class_id: number;
  target_class_name: string;
  class_names: string[];
  context_class_ids: number[];
  default_backend: string;
  imgsz: number;
  rank_metric: string;
  baselines_path: string;
  /** The profile a run uses when the request names none. */
  default?: boolean;
}

export interface BakeoffProfileList {
  profiles: BakeoffProfile[];
  count: number;
  /** Name of the `default: true` row; null when the configured default
   *  does not resolve (see `default_error`). */
  default_profile?: string | null;
  default_error?: string;
}

/** One failed piece of a bake-off: a whole stage (`throughput`, a
 *  quantize export) or one dataset × model cell. */
export interface BakeoffFailure {
  stage?: string;
  dataset?: string;
  model?: string;
  error: string;
}

/** `GET /bakeoff/status/{job_id}` — the runner's status.json. */
export interface BakeoffStatus {
  state?: 'enqueued' | 'running' | 'done' | 'error' | string;
  /** Job-level reason when `state` is `error`. */
  error?: string;
  progress?: { done: number; total: number };
  completed?: string[];
  failed?: BakeoffFailure[];
}

export function bakeoffProfiles(signal?: AbortSignal): Promise<BakeoffProfileList> {
  return apiFetch(`${API_PREFIX}/bakeoff/profiles`, {}, signal);
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
  return apiFetch(`${API_PREFIX}/bakeoff/trained_models`, {}, signal);
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
    /** BakeoffProfile name; omitted = the evaluator's deployment default. */
    profile?: string;
    verify_frozen?: boolean;
    job_id?: string;
  },
  signal?: AbortSignal,
): Promise<{ status: string; job_id: string; out_dir: string }> {
  return apiFetch(
    `${API_PREFIX}/bakeoff/run`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Auto-discovered frozen evaluation datasets (matrix columns). */
export function bakeoffEvalDatasets(
  signal?: AbortSignal,
): Promise<{ datasets: BakeoffEvalDataset[]; count: number }> {
  return apiFetch(`${API_PREFIX}/bakeoff/eval_datasets`, {}, signal);
}

/** Public/commercial baseline detectors from the editable registry —
 *  the given profile's own registry when it declares one. */
export function bakeoffBaselineModels(
  profile?: string,
  signal?: AbortSignal,
): Promise<{ baselines: BakeoffModelSpec[]; count: number }> {
  return apiFetch(`${API_PREFIX}/bakeoff/baseline_models${qs({ profile })}`, {}, signal);
}

/** The model x dataset matrix for a finished matrix job. */
export function bakeoffMatrix(
  jobId: string,
  signal?: AbortSignal,
): Promise<BakeoffMatrix> {
  return apiFetch(
    `${API_PREFIX}/bakeoff/matrix/${encodeURIComponent(jobId)}`,
    {},
    signal,
  );
}

export function bakeoffRuns(
  signal?: AbortSignal,
): Promise<{ runs: BakeoffRunSummary[] }> {
  return apiFetch(`${API_PREFIX}/bakeoff/runs`, {}, signal);
}

export function bakeoffStatus(
  jobId: string,
  signal?: AbortSignal,
): Promise<BakeoffStatus> {
  return apiFetch(
    `${API_PREFIX}/bakeoff/status/${encodeURIComponent(jobId)}`,
    {},
    signal,
  );
}

export function bakeoffResults(
  jobId: string,
  signal?: AbortSignal,
): Promise<BakeoffComparison> {
  return apiFetch(
    `${API_PREFIX}/bakeoff/results/${encodeURIComponent(jobId)}`,
    {},
    signal,
  );
}
