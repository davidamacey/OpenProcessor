/**
 * The cached 2-d embedding projection (`/viz/projection*`).
 */
import { apiFetch, isPlainObject, qs, scoped } from '$lib/api';
import { ApiError, structuredDetail } from '$lib/api_errors';

// -- embedding projection (2-d visualization overlay, Phase 5) -----------
//
// docs/curation-strategy-plan-2026-09.md §2.7/§5.6/§7 — `embedding_viz.py`
// + `viz.py` (OpenProcessor). UMAP-as-a-visualization-only overlay is the
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
  /** Not on the wire: `false` only for the served 404 `projection_not_built`. */
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
 * Fetch the cached 2-d projection (`ProjectionResponse`). Only the served
 * 404 `projection_not_built` means "nothing fit yet" (`built: false`); every
 * other failure, including a 404 without that code, a 503
 * `projection_unavailable` and a network error, rejects so the caller shows
 * the served message (`apiErrorText`) instead of a false "not built yet".
 */
export async function getVizProjection(
  params: {
    cluster_id?: number | null;
    class_id?: number | null;
    max_points?: number;
  } = {},
  signal?: AbortSignal,
): Promise<VizProjectionResponse> {
  let raw: unknown;
  try {
    raw = await apiFetch<unknown>(
      `${scoped()}/viz/projection${qs({
        cluster_id: params.cluster_id ?? undefined,
        class_id: params.class_id ?? undefined,
        max_points: params.max_points ?? undefined,
      })}`,
      {},
      signal,
    );
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) {
      const d = structuredDetail<{ error: string; message: string }>(e);
      if (d?.error === 'projection_not_built') return EMPTY_VIZ_PROJECTION;
    }
    throw e;
  }
  if (!isPlainObject(raw)) throw new Error('The projection response was not an object.');
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
 * the `buildRegionFpCentroids`/`clusterRegions` precedent — it lets the
 * error propagate for the caller to catch + toast rather than swallowing
 * it into a fallback value.
 */
export function rebuildVizProjection(signal?: AbortSignal): Promise<VizProjectionJob> {
  return apiFetch<VizProjectionJob>(
    `${scoped()}/viz/projection/rebuild`,
    { method: 'POST' },
    signal,
  );
}

/** Current rebuild-job snapshot — poll this after `rebuildVizProjection()`. */
export function getVizProjectionStatus(signal?: AbortSignal): Promise<VizProjectionJob> {
  return apiFetch<VizProjectionJob>(`${scoped()}/viz/projection/status`, {}, signal);
}

/** Cancel a running rebuild. `cancelled` is false when nothing was running. */
export function cancelVizProjection(
  signal?: AbortSignal,
): Promise<VizProjectionJob & { cancelled: boolean }> {
  return apiFetch<VizProjectionJob & { cancelled: boolean }>(
    `${scoped()}/viz/projection/cancel`,
    { method: 'POST' },
    signal,
  );
}
