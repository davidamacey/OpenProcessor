/**
 * The selectable assist runs (`{prefix}/assist/runs`): one request shape for a
 * crop, a selection, a box, a cluster, a sub-cluster or a filter. A request
 * with `dry_run` (the server's default) plans and writes nothing; the same
 * request with `dry_run: false` starts the run. Every count, refusal and
 * decision in the responses is the server's.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type {
  AssistApplyRequest,
  AssistApplyResponse,
  AssistRunList,
  AssistRunRequest,
  AssistRunState,
} from '$lib/types_assist';

export function createAssistRun(
  body: AssistRunRequest,
  signal?: AbortSignal,
): Promise<AssistRunState> {
  return apiFetch<AssistRunState>(
    `${scoped()}/assist/runs`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

export function getAssistRun(
  runId: string,
  signal?: AbortSignal,
): Promise<AssistRunState> {
  return apiFetch<AssistRunState>(
    `${scoped()}/assist/runs/${encodeURIComponent(runId)}`,
    {},
    signal,
  );
}

/** The newest runs (newest first) that name this cluster, each with `scope_now`. */
export function listAssistRuns(
  scope: { clusterId?: number; clusterSubid?: string | null; limit?: number } = {},
  signal?: AbortSignal,
): Promise<AssistRunList> {
  return apiFetch<AssistRunList>(
    `${scoped()}/assist/runs${qs({
      cluster_id: scope.clusterId,
      cluster_subid: scope.clusterSubid ?? undefined,
      limit: scope.limit,
    })}`,
    {},
    signal,
  );
}

export function cancelAssistRun(
  runId: string,
  signal?: AbortSignal,
): Promise<AssistRunState> {
  return apiFetch<AssistRunState>(
    `${scoped()}/assist/runs/${encodeURIComponent(runId)}/cancel`,
    { method: 'POST' },
    signal,
  );
}

export function applyAssistRun(
  runId: string,
  body: AssistApplyRequest,
  signal?: AbortSignal,
): Promise<AssistApplyResponse> {
  return apiFetch<AssistApplyResponse>(
    `${scoped()}/assist/runs/${encodeURIComponent(runId)}/apply`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}
