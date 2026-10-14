/**
 * Pool-scale diverse selection (`/select*`) and the selection writes that act
 * on every item a filter matches.
 */
import { apiFetch, isPlainObject, scoped } from '$lib/api';
import { ApiError } from '$lib/api_errors';
import type { ExcludeReason } from '$lib/api_cropWrites';
import type { BulkLabelResult } from '$lib/types';
import type {
  DiverseSelection,
  SelectDiverseScope,
  SelectJobStatus,
} from '$lib/types_review';
import type {
  ItemSelection,
  SelectionDryRun,
  SelectionExcludeResult,
} from '$lib/types_itemFilter';

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
      `${scoped()}/select/diverse`,
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
  const raw = await apiFetch<unknown>(`${scoped()}/select/status`, {}, signal);
  return parseSelectJobStatus(raw);
}

/** Cancel the singleton diverse-selection job, if any is running. Real
 *  backend returns `{cancelled: bool, ...job state}` (select.py's
 *  `select_cancel`), not a bare 204 — the caller only needs to know
 *  polling can stop, so the body is discarded. */
export async function cancelSelect(signal?: AbortSignal): Promise<void> {
  await apiFetch<unknown>(`${scoped()}/select/cancel`, { method: 'POST' }, signal);
}

// -- selection writes: act on every item a filter matches ----------------
//
// The four writes below take `selection` (an `ItemSelection`: a filter,
// optionally limited / sampled) in place of `crop_ids`, plus `dry_run`.
// A dry run answers `{dry_run: true, selected}` and writes nothing; the
// real call returns the same body the crop-id form does.

export type SelectionWrite<T> = T | SelectionDryRun;

export function bulkLabelSelection(
  selection: ItemSelection,
  classId: number,
  dryRun: boolean,
  signal?: AbortSignal,
): Promise<SelectionWrite<BulkLabelResult>> {
  return apiFetch(
    `${scoped()}/crops/batch_label`,
    {
      method: 'PUT',
      body: JSON.stringify({
        selection,
        class_id: classId,
        validated: true,
        dry_run: dryRun,
      }),
    },
    signal,
  );
}

export function batchExcludeSelection(
  selection: ItemSelection,
  reason: ExcludeReason,
  dryRun: boolean,
  signal?: AbortSignal,
): Promise<SelectionWrite<SelectionExcludeResult & { excluded: number }>> {
  return apiFetch(
    `${scoped()}/crops/batch_exclude`,
    { method: 'POST', body: JSON.stringify({ selection, reason, dry_run: dryRun }) },
    signal,
  );
}

/** Restoring ignored items means selecting excluded ones, hence
 *  `include_excluded`. */
export function batchUnexcludeSelection(
  selection: ItemSelection,
  dryRun: boolean,
  signal?: AbortSignal,
): Promise<SelectionWrite<SelectionExcludeResult & { unexcluded: number }>> {
  return apiFetch(
    `${scoped()}/crops/batch_unexclude`,
    {
      method: 'POST',
      body: JSON.stringify({
        selection: { ...selection, include_excluded: true },
        dry_run: dryRun,
      }),
    },
    signal,
  );
}

export function moveSelectionToCluster(
  selection: ItemSelection,
  clusterId: number,
  dryRun: boolean,
  signal?: AbortSignal,
): Promise<SelectionWrite<BulkLabelResult>> {
  return apiFetch(
    `${scoped()}/crops/move`,
    {
      method: 'POST',
      body: JSON.stringify({ selection, cluster_id: clusterId, dry_run: dryRun }),
    },
    signal,
  );
}
