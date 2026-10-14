/**
 * Region profiles, the segmenter-prompt check and the config vocabulary
 * (`/config/region_profiles*`).
 */
import { apiFetch, qs, scoped } from '$lib/api';
import { structuredDetail } from '$lib/api_errors';
import type {
  ActivationImpact,
  ConfigVocabulary,
  ProfileActivateResponse,
  ProfileUpdateRequest,
  ProfileValidateRequest,
  RegionProfileDoc,
  RegionProfileList,
  RegionProfileSchema,
  SegmenterPromptValidateRequest,
} from '$lib/types_profiles';
import type {
  ActiveConfigResponse,
  ActiveRef,
  ConfigActivateRequest,
  ConfigCloneRequest,
  ConfigErrorDetail,
  ConfigRevisionList,
  ValidationReport,
} from '$lib/types_config';

// -- Region profiles and the config vocabulary ---------------------------

/** `GET /region_profiles?include_templates=true`: every profile (env,
 *  registered, stored) plus the clone-only templates and the active ref.
 *  Also the profile gate's probe. */
export function listRegionProfiles(signal?: AbortSignal): Promise<RegionProfileList> {
  return apiFetch<RegionProfileList>(
    `${scoped()}/region_profiles${qs({ include_templates: true })}`,
    {},
    signal,
  );
}

export function getRegionProfileSchema(
  signal?: AbortSignal,
): Promise<RegionProfileSchema> {
  return apiFetch<RegionProfileSchema>(`${scoped()}/region_profiles/schema`, {}, signal);
}

/** `POST /region_profiles/validate`: a draft's report, never a write.
 *  `forActivation` adds the activation-only checks (§4.3). */
export function validateRegionProfile(
  body: ProfileValidateRequest,
  forActivation: boolean,
  signal?: AbortSignal,
): Promise<ValidationReport> {
  return apiFetch<ValidationReport>(
    `${scoped()}/region_profiles/validate${qs({ for_activation: forActivation })}`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** `POST /region_profiles/validate_segmenter_prompt`: the served text-only
 *  check of a segmenter prompt. */
export function validateSegmenterPrompt(
  body: SegmenterPromptValidateRequest,
  signal?: AbortSignal,
): Promise<ValidationReport> {
  return apiFetch<ValidationReport>(
    `${scoped()}/region_profiles/validate_segmenter_prompt`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

export function getRegionProfile(
  name: string,
  signal?: AbortSignal,
): Promise<RegionProfileDoc> {
  return apiFetch<RegionProfileDoc>(
    `${scoped()}/region_profiles/${encodeURIComponent(name)}`,
    {},
    signal,
  );
}

export function getRegionProfileRevisions(
  name: string,
  signal?: AbortSignal,
): Promise<ConfigRevisionList> {
  return apiFetch<ConfigRevisionList>(
    `${scoped()}/region_profiles/${encodeURIComponent(name)}/revisions`,
    {},
    signal,
  );
}

export function getRegionProfileRevision(
  name: string,
  revision: number,
  signal?: AbortSignal,
): Promise<RegionProfileDoc> {
  return apiFetch<RegionProfileDoc>(
    `${scoped()}/region_profiles/${encodeURIComponent(name)}/revisions/${encodeURIComponent(String(revision))}`,
    {},
    signal,
  );
}

/** `POST /region_profiles/{name}/clone` → 201 the new stored profile. */
export function cloneRegionProfile(
  name: string,
  body: ConfigCloneRequest,
): Promise<RegionProfileDoc> {
  return apiFetch<RegionProfileDoc>(
    `${scoped()}/region_profiles/${encodeURIComponent(name)}/clone`,
    { method: 'POST', body: JSON.stringify(body) },
  );
}

/** `PUT /region_profiles/{name}`: saves a new revision. Never changes what
 *  runs (§4.4); 409 `revision_conflict` carries the current revision. */
export function updateRegionProfile(
  name: string,
  body: ProfileUpdateRequest,
): Promise<RegionProfileDoc> {
  return apiFetch<RegionProfileDoc>(
    `${scoped()}/region_profiles/${encodeURIComponent(name)}`,
    { method: 'PUT', body: JSON.stringify(body) },
  );
}

/** `DELETE /region_profiles/{name}?expected_revision=` → 204 (409 `in_use`
 *  when it is the active profile). */
export function deleteRegionProfile(
  name: string,
  expectedRevision: number,
): Promise<void> {
  return apiFetch<void>(
    `${scoped()}/region_profiles/${encodeURIComponent(name)}${qs({ expected_revision: expectedRevision })}`,
    { method: 'DELETE' },
  );
}

/** `GET /region_profiles/active` (axis `detection_profile`; `active.name`
 *  null = region detection is off). */
export function getActiveRegionProfile(
  signal?: AbortSignal,
): Promise<ActiveConfigResponse> {
  return apiFetch<ActiveConfigResponse>(`${scoped()}/region_profiles/active`, {}, signal);
}

/** `POST /region_profiles/{name}/activate` (OCC on `expected_active`); the
 *  response adds the served `impact` and `validation`. */
export function activateRegionProfile(
  name: string,
  body: ConfigActivateRequest,
): Promise<ProfileActivateResponse> {
  return apiFetch<ProfileActivateResponse>(
    `${scoped()}/region_profiles/${encodeURIComponent(name)}/activate`,
    { method: 'POST', body: JSON.stringify(body) },
  );
}

/** `POST /region_profiles/active/rollback`: re-activates the previous one. */
export function rollbackRegionProfile(body: {
  expected_active: ActiveRef;
}): Promise<ActiveConfigResponse> {
  return apiFetch<ActiveConfigResponse>(`${scoped()}/region_profiles/active/rollback`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

/** `POST /region_profiles/deactivate`: region detection off (OCC). */
export function deactivateRegionProfile(body: {
  expected_active: ActiveRef;
}): Promise<ActiveConfigResponse> {
  return apiFetch<ActiveConfigResponse>(`${scoped()}/region_profiles/deactivate`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

/** `GET /region_profiles/active/impact`: items by the profile@revision that
 *  produced them, plus the served re-run suggestion (§4.6). */
export function getRegionProfileImpact(signal?: AbortSignal): Promise<ActivationImpact> {
  return apiFetch<ActivationImpact>(
    `${scoped()}/region_profiles/active/impact`,
    {},
    signal,
  );
}

/** `GET /config/vocabulary`: every model / mode / class list the profile
 *  editor's pickers render (§7.4). `includeOtherProjects` adds other
 *  projects' shared detectors (projects_plan.md §5.5). */
export function getConfigVocabulary(
  includeOtherProjects: boolean,
  signal?: AbortSignal,
): Promise<ConfigVocabulary> {
  return apiFetch<ConfigVocabulary>(
    `${scoped()}/config/vocabulary${qs({ include_other_projects: includeOtherProjects || undefined })}`,
    {},
    signal,
  );
}

/** The structured config-store refusal (`{detail: ConfigErrorDetail}`,
 *  §7.1) of a prompt-pack or region-profile route, or `null` when the
 *  error isn't one. The UI shows `message`, branches on `error`. */
export const configErrorDetail = (e: unknown): ConfigErrorDetail | null =>
  structuredDetail<ConfigErrorDetail>(e);
