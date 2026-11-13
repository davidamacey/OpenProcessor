/**
 * Data quality reads. Types live in `$lib/types_quality`.
 */
import { apiFetch, scoped } from '$lib/api';
import type { DatasetReadiness } from '$lib/types_quality';

/** `GET /quality/readiness`: the dataset-level "ready to train" checklist, one row per check
 *  with its served label, severity, message and link. */
export function getReadiness(signal?: AbortSignal): Promise<DatasetReadiness> {
  return apiFetch<DatasetReadiness>(`${scoped()}/quality/readiness`, {}, signal);
}
