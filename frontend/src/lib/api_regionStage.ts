/**
 * Region-stage wrappers (`{prefix}/region_stage` and its `/pause` /
 * `/resume` writes, OpenProcessor v0.4.0). Called only while a region
 * profile is served (without one the routes answer 409). Import from
 * `$lib/api_regionStage` directly; never re-exported from `api.ts` (that
 * would make the two modules circular).
 */
import { apiFetch, scoped } from '$lib/api';
import type { RegionStageState } from '$lib/types_openVocab';

export function getRegionStage(signal?: AbortSignal): Promise<RegionStageState> {
  return apiFetch<RegionStageState>(`${scoped()}/region_stage`, {}, signal);
}

/** `POST /region_stage/pause`: queued items stay pending. */
export function pauseRegionStage(): Promise<RegionStageState> {
  return apiFetch<RegionStageState>(`${scoped()}/region_stage/pause`, {
    method: 'POST',
  });
}

/** `POST /region_stage/resume`. */
export function resumeRegionStage(): Promise<RegionStageState> {
  return apiFetch<RegionStageState>(`${scoped()}/region_stage/resume`, {
    method: 'POST',
  });
}
