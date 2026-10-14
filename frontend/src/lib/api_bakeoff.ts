/**
 * Bake-off profiles, runs, results and the comparison matrix (`/bakeoff/*`).
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type {
  BakeoffComparison,
  BakeoffMatrix,
  BakeoffProfileList,
  BakeoffRunAccepted,
  BakeoffRunList,
  BakeoffRunRequest,
  BakeoffStatus,
  BaselineModelList,
  EvalDatasetList,
  TrainedModelList,
} from '$lib/types_bakeoff';

// ===========================================================================
// Model comparison ({API_PREFIX}/bakeoff) — v2 wire, OpenProcessor #34 §7.
// Types live in `./types_bakeoff.ts`.
// ===========================================================================

export function bakeoffProfiles(signal?: AbortSignal): Promise<BakeoffProfileList> {
  return apiFetch(`${scoped()}/bakeoff/profiles`, {}, signal);
}

/** The given profile's baseline registry (empty by default). */
export function bakeoffBaselineModels(
  profile?: string,
  signal?: AbortSignal,
): Promise<BaselineModelList> {
  return apiFetch(`${scoped()}/bakeoff/baseline_models${qs({ profile })}`, {}, signal);
}

/** Export test splits and external frozen sets, in served order. */
export function bakeoffEvalDatasets(
  source?: 'export' | 'external',
  signal?: AbortSignal,
): Promise<EvalDatasetList> {
  return apiFetch(`${scoped()}/bakeoff/eval_datasets${qs({ source })}`, {}, signal);
}

/** Finished training runs; with `datasetId`, each carries `for_dataset`. */
export function bakeoffTrainedModels(
  params: { datasetId?: string; limit?: number } = {},
  signal?: AbortSignal,
): Promise<TrainedModelList> {
  return apiFetch(
    `${scoped()}/bakeoff/trained_models${qs({ dataset_id: params.datasetId, limit: params.limit })}`,
    {},
    signal,
  );
}

export function bakeoffRun(
  body: BakeoffRunRequest,
  signal?: AbortSignal,
): Promise<BakeoffRunAccepted> {
  return apiFetch(
    `${scoped()}/bakeoff/run`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

export function bakeoffStatus(
  jobId: string,
  signal?: AbortSignal,
): Promise<BakeoffStatus> {
  return apiFetch(`${scoped()}/bakeoff/status/${encodeURIComponent(jobId)}`, {}, signal);
}

export function bakeoffRuns(signal?: AbortSignal): Promise<BakeoffRunList> {
  return apiFetch(`${scoped()}/bakeoff/runs`, {}, signal);
}

/** One dataset's comparison (default: the job's first dataset). A 409
 *  means the stored result predates the v2 format. */
export function bakeoffResults(
  jobId: string,
  datasetId?: string,
  signal?: AbortSignal,
): Promise<BakeoffComparison> {
  return apiFetch(
    `${scoped()}/bakeoff/results/${encodeURIComponent(jobId)}${qs({ dataset_id: datasetId })}`,
    {},
    signal,
  );
}

export function bakeoffMatrix(
  jobId: string,
  signal?: AbortSignal,
): Promise<BakeoffMatrix> {
  return apiFetch(`${scoped()}/bakeoff/matrix/${encodeURIComponent(jobId)}`, {}, signal);
}
