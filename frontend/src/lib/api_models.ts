/**
 * Model status, cross-project sharing, class mapping and unload
 * (`/models*`).
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type {
  ModelsStatus,
  UnloadModelResponse,
  ModelClassMappingResponse,
  ModelSharingRequest,
  ModelSharingResponse,
} from '$lib/types_models';

/**
 * Every model the active project can see: its own, the base models, and
 * (projects P2, §5.5) other projects' models their owners shared, each
 * with the served `project`/`shared`/`class_mapping`.
 */
export function getModelsStatus(signal?: AbortSignal): Promise<ModelsStatus> {
  return apiFetch<ModelsStatus>(
    `${scoped()}/models/status${qs({ include_other_projects: true })}`,
    {},
    signal,
  );
}

/**
 * Opt one of the active project's promoted models into (or out of)
 * cross-project sharing. Owner only: any other project gets 404
 * `model_not_found`. 409 `revision_conflict` carries the served
 * `current_revision`; 409 `in_use` (unsharing while another project uses
 * it) is bypassed by `force`.
 */
export function setModelSharing(
  modelName: string,
  body: ModelSharingRequest,
  force = false,
  signal?: AbortSignal,
): Promise<ModelSharingResponse> {
  return apiFetch<ModelSharingResponse>(
    `${scoped()}/models/${encodeURIComponent(modelName)}/sharing${qs({ force: force || undefined })}`,
    { method: 'PUT', body: JSON.stringify(body) },
    signal,
  );
}

/** How a model's classes map by name onto the active project's registry. */
export function getModelClassMapping(
  modelName: string,
  signal?: AbortSignal,
): Promise<ModelClassMappingResponse> {
  return apiFetch<ModelClassMappingResponse>(
    `${scoped()}/models/${encodeURIComponent(modelName)}/class_mapping`,
    {},
    signal,
  );
}

/**
 * Unload a Triton model and remove its repo directory (follow-up gap 2,
 * docs/design/audit-remediation-plan-2026-09.md Appendix D item 3,
 * 2026-09-11). The backend enforces the real guard (region-protected
 * models never, active/core models need `force`) — `force` here only
 * matters for the latter; passing it for a region-protected model still
 * 403s.
 */
export function unloadModel(
  modelName: string,
  force = false,
  signal?: AbortSignal,
): Promise<UnloadModelResponse> {
  return apiFetch<UnloadModelResponse>(
    `${scoped()}/models/${encodeURIComponent(modelName)}${qs({ force })}`,
    { method: 'DELETE' },
    signal,
  );
}
