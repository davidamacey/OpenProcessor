/**
 * Curation scores: field coverage and the compute job (`/scores*`).
 */
import { apiFetch, isPlainObject, scoped } from '$lib/api';

// -- curation scores (`/settings` "Curation scores" card, G10) -----------
//
// docs/design/frontend-coverage-audit-2026-09-24.md §G10: review queues
// (Uncertainty, Model Disagreements, and the `uniqueness`/`mistakenness`
// StrategyBar sorts) are empty not because nothing's wrong but because
// no scorer has ever run — `/scores/*` had no frontend caller at all.
// Scorers stay operator-triggered by design (OpenProcessor's
// `src/routers/curation/scores.py` docstring); this is a deployment-level operation, so it
// lives on `/settings`, not a StrategyBar chip, following the same
// confirm-before-write convention as the rest of that page.
//
// Wire shapes confirmed 2026-09-24 against OpenProcessor `main`'s scores
// job module (`compute_coverage`/`_JobState`) — the vendored OpenAPI
// spec only declares
// `additionalProperties: true` for these four routes, so there is
// nothing to check against `endpointCatalog.test.ts` beyond path+method.

/** One scorer's coverage row, keyed by scorer id in `ScoresCoverage`. */
export interface ScoreCoverageEntry {
  /** The OpenSearch field this scorer writes — an `exists` count on this
   *  field is what `n_scored` counts. */
  field: string;
  n_scored: number;
  total: number;
  pct: number;
}

/** `GET {API_PREFIX}/scores/coverage`'s `coverage` map — scorer id → row.
 *  Scorer ids are never hardcoded frontend-side; they come from this
 *  map's own keys (`Object.keys`), same rule the compute request body
 *  follows. */
export type ScoresCoverage = Record<string, ScoreCoverageEntry>;

function parseScoreCoverageEntry(raw: unknown): ScoreCoverageEntry | null {
  if (!isPlainObject(raw)) return null;
  const { field, n_scored, total, pct } = raw;
  if (typeof field !== 'string') return null;
  if (typeof n_scored !== 'number') return null;
  if (typeof total !== 'number') return null;
  if (typeof pct !== 'number') return null;
  return { field, n_scored, total, pct };
}

function parseScoresCoverage(raw: unknown): ScoresCoverage {
  if (!isPlainObject(raw) || !isPlainObject(raw.coverage)) return {};
  const out: ScoresCoverage = {};
  for (const [scorerId, entryRaw] of Object.entries(raw.coverage)) {
    const entry = parseScoreCoverageEntry(entryRaw);
    if (entry) out[scorerId] = entry;
  }
  return out;
}

/**
 * Per-scorer coverage. Rejects on failure; the one caller, the
 * `/settings` scores card, shows the error with a retry.
 */
export async function getScoresCoverage(signal?: AbortSignal): Promise<ScoresCoverage> {
  const raw = await apiFetch<unknown>(`${scoped()}/scores/coverage`, {}, signal);
  return parseScoresCoverage(raw);
}

/** `POST /scores/compute` / `GET /scores/status` / `POST /scores/cancel`
 *  job snapshot — CONFIRMED against `crop_scores/job.py`'s `_JobState`.
 *  `total`/`processed` count *crops in the shared embedding fetch*, not
 *  scorers — every enabled scorer in `scorers` runs against the same
 *  fetched matrix (job.py's "single-fetch design"), so there is no
 *  meaningful per-scorer progress split to show. `error` is the raw
 *  server string verbatim — never reworded — because it is frequently
 *  the actionable detail (e.g. mistakenness failing for lack of probe
 *  predictions, per the backend note this card's spec was written
 *  against). */
export interface ScoresJob {
  job_id: string;
  status: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';
  scorers: string[];
  processed: number;
  total: number;
  started_at: number;
  finished_at: number;
  error: string | null;
  results: Record<string, unknown>;
}

const IDLE_SCORES_JOB: ScoresJob = {
  job_id: '',
  status: 'idle',
  scorers: [],
  processed: 0,
  total: 0,
  started_at: 0,
  finished_at: 0,
  error: null,
  results: {},
};

function parseScoresJob(raw: unknown): ScoresJob {
  if (!isPlainObject(raw)) return { ...IDLE_SCORES_JOB };
  const status = raw.status;
  const validStatus =
    status === 'idle' ||
    status === 'running' ||
    status === 'completed' ||
    status === 'failed' ||
    status === 'cancelled';
  return {
    job_id: typeof raw.job_id === 'string' ? raw.job_id : '',
    status: validStatus ? status : 'idle',
    scorers: Array.isArray(raw.scorers)
      ? raw.scorers.filter((s): s is string => typeof s === 'string')
      : [],
    processed: typeof raw.processed === 'number' ? raw.processed : 0,
    total: typeof raw.total === 'number' ? raw.total : 0,
    started_at: typeof raw.started_at === 'number' ? raw.started_at : 0,
    finished_at: typeof raw.finished_at === 'number' ? raw.finished_at : 0,
    error: typeof raw.error === 'string' ? raw.error : null,
    results: isPlainObject(raw.results) ? raw.results : {},
  };
}

/**
 * Kick off (or resume) a scoring run. `scorers: null` runs every scorer
 * the backend currently enables (`ScoresComputeRequest.scorers`'
 * documented `None` meaning) — the card's "Compute all" action sends
 * `null`, never a client-enumerated id list, so a scorer the frontend
 * doesn't know about yet still gets included. Rejects on failure
 * (a 400 "scoring disabled"/"unknown scorer" or 409 "already running" —
 * `ApiError.detail` carries the server's own text verbatim) so the
 * caller can show it rather than swallow it, matching
 * `rebuildVizProjection`'s explicit-user-action contract.
 */
export function computeScores(
  scorers: string[] | null,
  signal?: AbortSignal,
): Promise<ScoresJob> {
  return apiFetch<unknown>(
    `${scoped()}/scores/compute`,
    { method: 'POST', body: JSON.stringify({ scorers }) },
    signal,
  ).then(parseScoresJob);
}

/** Current/last scoring-job snapshot — poll this after `computeScores()`. */
export function getScoresStatus(signal?: AbortSignal): Promise<ScoresJob> {
  return apiFetch<unknown>(`${scoped()}/scores/status`, {}, signal).then(parseScoresJob);
}

/** Cancel a running scoring job. Real backend returns `{cancelled, ...job
 *  state}` (`src/routers/curation/scores.py::scores_cancel`), same shape as
 *  `cancelVizProjection` — `cancelled` is false when nothing was running. */
export async function cancelScores(
  signal?: AbortSignal,
): Promise<ScoresJob & { cancelled: boolean }> {
  const raw = await apiFetch<unknown>(
    `${scoped()}/scores/cancel`,
    { method: 'POST' },
    signal,
  );
  const job = parseScoresJob(raw);
  const cancelled = isPlainObject(raw) && raw.cancelled === true;
  return { ...job, cancelled };
}
