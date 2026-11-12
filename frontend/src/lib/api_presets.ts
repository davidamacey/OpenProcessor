/**
 * Research preset wrappers: the six routes each kind has (`list`, `get`, `put`, `delete`, `activate`
 * and `active`) under `/serving_profiles` (deployment-wide) and `/cluster_profiles`,
 * `/assist_loop_presets`, `/near_dup_profiles` (per project). Import from `$lib/api_presets`
 * directly. Callers render `apiErrorText`; a refused save also carries `reasons`
 * (`presetRefusalText`).
 */
import { apiFetch, globalApi, qs, scoped } from '$lib/api';
import { ApiError, apiErrorText } from '$lib/api_errors';
import { configErrorDetail } from '$lib/api_profiles';
import type {
  ActivePreset,
  PresetActivateRequest,
  PresetActiveWire,
  PresetDoc,
  PresetKindId,
  PresetListResponse,
  PresetPutRequest,
  PresetSchemaWire,
  ServingActivateRequest,
  ServingActiveWire,
} from '$lib/types_presets';

const GLOBAL = { global: true } as const;

/** The collection URL of a kind. One literal per kind: the endpoint catalog reads them. */
function collection(kind: PresetKindId): string {
  switch (kind) {
    case 'serving_profile':
      return `${globalApi()}/serving_profiles`;
    case 'cluster_profile':
      return `${scoped()}/cluster_profiles`;
    case 'assist_loop_preset':
      return `${scoped()}/assist_loop_presets`;
    case 'near_dup_profile':
      return `${scoped()}/near_dup_profiles`;
  }
}

const opts = (kind: PresetKindId) => (kind === 'serving_profile' ? GLOBAL : {});

const item = (kind: PresetKindId, name: string) =>
  `${collection(kind)}/${encodeURIComponent(name)}`;

export function listPresets(
  kind: PresetKindId,
  signal?: AbortSignal,
): Promise<PresetListResponse> {
  return apiFetch<PresetListResponse>(collection(kind), {}, signal, opts(kind));
}

/** `GET /schema`: the JSON Schema of the kind's body, for the field help beside the editor. */
export function getPresetSchema(
  kind: PresetKindId,
  signal?: AbortSignal,
): Promise<PresetSchemaWire> {
  return apiFetch<PresetSchemaWire>(`${collection(kind)}/schema`, {}, signal, opts(kind));
}

/** `GET /{name}`; `revision` reads an older one. */
export function getPreset(
  kind: PresetKindId,
  name: string,
  revision?: number,
  signal?: AbortSignal,
): Promise<PresetDoc> {
  return apiFetch<PresetDoc>(
    `${item(kind, name)}${qs({ revision })}`,
    {},
    signal,
    opts(kind),
  );
}

/** `PUT /{name}`: creates (`expected_revision: null`) or saves the next revision. */
export function putPreset(
  kind: PresetKindId,
  name: string,
  req: PresetPutRequest,
): Promise<PresetDoc> {
  return apiFetch<PresetDoc>(
    item(kind, name),
    { method: 'PUT', body: JSON.stringify(req) },
    undefined,
    opts(kind),
  );
}

/** `DELETE /{name}?expected_revision=` -> 204. */
export function deletePreset(
  kind: PresetKindId,
  name: string,
  expectedRevision: number,
): Promise<void> {
  return apiFetch<void>(
    `${item(kind, name)}${qs({ expected_revision: expectedRevision })}`,
    { method: 'DELETE' },
    undefined,
    opts(kind),
  );
}

const NONE: ActivePreset = {
  axis: null,
  state: 'none',
  name: null,
  revision: null,
  activated_at: null,
  setting_revision: null,
};

/** `GET /active?axis=` of a project kind: what is active on one axis. */
export async function getActivePreset(
  kind: Exclude<PresetKindId, 'serving_profile'>,
  axis: string,
  signal?: AbortSignal,
): Promise<ActivePreset> {
  const w = await apiFetch<PresetActiveWire>(
    `${collection(kind)}/active${qs({ axis })}`,
    {},
    signal,
  );
  return {
    axis: w.axis,
    state: w.state,
    name: w.name ?? null,
    revision: w.revision ?? null,
    activated_at: w.activated_at ?? null,
    setting_revision: null,
  };
}

/** `GET /serving_profiles/active`: a 404 `not_found` is "none active". */
export async function getActiveServingProfile(
  signal?: AbortSignal,
): Promise<ActivePreset> {
  try {
    const w = await apiFetch<ServingActiveWire>(
      `${collection('serving_profile')}/active`,
      {},
      signal,
      GLOBAL,
    );
    return {
      axis: null,
      state: 'active',
      name: w.name,
      revision: w.revision,
      activated_at: null,
      setting_revision: w.setting_revision,
    };
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return NONE;
    throw e;
  }
}

/** `POST /{name}/activate` of a project kind. */
export function activatePreset(
  kind: Exclude<PresetKindId, 'serving_profile'>,
  name: string,
  req: PresetActivateRequest,
): Promise<PresetActiveWire> {
  return apiFetch<PresetActiveWire>(`${item(kind, name)}/activate`, {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

/** `POST /serving_profiles/{name}/activate`. */
export function activateServingProfile(
  name: string,
  req: ServingActivateRequest,
): Promise<ServingActiveWire> {
  return apiFetch<ServingActiveWire>(
    `${item('serving_profile', name)}/activate`,
    { method: 'POST', body: JSON.stringify(req) },
    undefined,
    GLOBAL,
  );
}

/** The served message of a refusal followed by its `reasons` (a 422 on Save), as one text. */
export function presetRefusalText(e: unknown): string {
  const reasons = configErrorDetail(e)?.reasons ?? [];
  return reasons.length > 0
    ? `${apiErrorText(e)}: ${reasons.join('; ')}`
    : apiErrorText(e);
}
