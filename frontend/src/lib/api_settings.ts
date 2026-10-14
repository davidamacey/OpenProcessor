/**
 * Strategy capabilities (`/methods`), the shared curation defaults
 * (`/settings`) and the per-project keymap (`/keymap`).
 */
import { apiFetch, scoped } from '$lib/api';
import { ApiError } from '$lib/api_errors';
import { type MethodsResponse, parseMethodsResponse } from '$lib/strategies';
import { type CurationSettings, parseCurationSettings } from '$lib/curationSettings';
import type { KeymapDocument, KeymapValidationIssue } from '$lib/keymapFallback';

/**
 * Capability discovery for the curation-strategy registries: which
 * cluster methods / review sorts / overlays / scores / exports / assist
 * axes the backend currently offers, each with a `stable | experimental
 * | shadow | disabled` status. Rejects on failure like every other read;
 * `strategiesStore` is the one caller that catches.
 */
export async function getMethods(signal?: AbortSignal): Promise<MethodsResponse> {
  const raw = await apiFetch<unknown>(`${scoped()}/methods`, {}, signal);
  return parseMethodsResponse(raw);
}

// -- shared curation defaults (GET,PUT {API_PREFIX}/settings) -----------
//
// Deployment-wide strategy defaults, verified against wt-oss-hardening's
// `src/routers/curation/settings.py` on 2026-09-21. See
// docs/design/curation-settings-ui-plan-2026-09-21.md §1.3.

/**
 * Read the deployment's shared curation defaults. A `defaults: {}`
 * record (nothing written yet) is the normal first-run response, not an
 * error. Rejects on failure; `curationSettingsStore` is the one caller
 * that catches.
 */
export async function getCurationSettings(
  signal?: AbortSignal,
): Promise<CurationSettings> {
  const raw = await apiFetch<unknown>(`${scoped()}/settings`, {}, signal);
  return parseCurationSettings(raw);
}

/**
 * Merge one or more axis defaults into the shared record.
 *
 * PARTIAL BODY BY CONTRACT — send only the axes being changed. The
 * backend writes `{'doc': {...}, 'doc_as_upsert': True}`, an OpenSearch
 * recursive object merge, so axes not mentioned are left untouched. This
 * is also what protects an axis id this build has never heard of from
 * being clobbered by an older frontend: never send the whole `defaults`
 * map back, only the delta.
 *
 * Throws `ApiError` with `status === 422` when an axis is unsettable or
 * an id is not currently advertised for it; `ApiError.detail` carries the
 * server's own message listing the valid axes/ids. Note `errorDetail()`
 * truncates at 200 chars, so a long `valid ids: [...]` list can be
 * elided — the caller should re-sync `/methods` rather than rely on
 * parsing that string (see the store's `saveDefault`).
 *
 * A `null` value for an axis CLEARS that axis's shared override — the
 * backend drops it from the stored record entirely, so the next GET's
 * `defaults` map omits that key and callers fall back to the axis's own
 * built-in default. This is the fix for the gap once tracked as H-1 in
 * docs/design/curation-settings-ui-plan-2026-09-21.md (a pinned default
 * used to be permanent, since PUT only merged and every value had to be
 * a currently-advertised id).
 *
 * Returns the FULL merged record. Always adopt this; never
 * optimistically construct the post-save state client-side.
 */
export async function putCurationDefaults(
  defaults: Record<string, string | null>,
  signal?: AbortSignal,
): Promise<CurationSettings> {
  const raw = await apiFetch<unknown>(
    `${scoped()}/settings`,
    { method: 'PUT', body: JSON.stringify({ defaults }) },
    signal,
  );
  return parseCurationSettings(raw);
}

// -- configurable keyboard shortcuts (K2, docs/design/
//    configurable-keyboard-shortcuts-plan-2026-09-26.md §0/§4) ----------
//
// The keymap is per-project (owner decision §0.1): every route below is
// scoped through `scoped()`, and there is no separate global route. A
// backend that predates the keymap route 404s/501s `GET {prefix}/keymap`
// — `keymapAvailability`/`loadKeymap()` treat that as "stay on
// FALLBACK_KEYMAP", not an error.

export interface KeymapValidateRequest {
  overrides: Record<string, string[]>;
}

export interface KeymapClassConflict {
  project: string;
  class_id: number;
  class_name: string;
  combo: string;
  action_id: string;
}

export interface KeymapValidationReport {
  ok: boolean;
  errors: KeymapValidationIssue[];
  warnings: KeymapValidationIssue[];
  force_allowed: boolean;
  resolved?: Record<string, string[]>;
  reserved_hotkeys?: string[];
  class_conflicts?: KeymapClassConflict[];
}

export interface KeymapPutRequest {
  expected_revision: number;
  overrides: Record<string, string[]>;
  unbind_conflicting_class_hotkeys?: boolean;
}

export interface KeymapPutResponse extends KeymapDocument {
  unbound_class_hotkeys?: Array<{
    project: string;
    class_id: number;
    class_name: string;
    was: string;
  }>;
}

export interface KeymapResetRequest {
  expected_revision: number;
  /** `null`/omitted = reset every action. */
  action_ids?: string[] | null;
  unbind_conflicting_class_hotkeys?: boolean;
}

/** `GET {prefix}/keymap` — the effective document for the active project. */
export async function getKeymap(signal?: AbortSignal): Promise<KeymapDocument> {
  return apiFetch<KeymapDocument>(`${scoped()}/keymap`, {}, signal);
}

/** `POST {prefix}/keymap/validate` — always 200, a dry-run report. Never
 *  throws on `ok: false`; the caller renders `errors`/`warnings` verbatim. */
export async function validateKeymap(
  body: KeymapValidateRequest,
  signal?: AbortSignal,
): Promise<KeymapValidationReport> {
  return apiFetch<KeymapValidationReport>(
    `${scoped()}/keymap/validate`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/**
 * `PUT {prefix}/keymap` — replace the whole override map (OCC via
 * `expected_revision`).
 *
 * Throws `ApiError` on:
 *   409 `revision_conflict`      — `keymapRevisionConflictDetail(e)`
 *   409 `class_hotkey_conflict`  — `keymapClassConflictDetail(e)`
 *   422 `validation_failed`      — `keymapValidationFailedDetail(e)`
 */
export async function putKeymap(
  body: KeymapPutRequest,
  signal?: AbortSignal,
): Promise<KeymapPutResponse> {
  return apiFetch<KeymapPutResponse>(
    `${scoped()}/keymap`,
    { method: 'PUT', body: JSON.stringify(body) },
    signal,
  );
}

/** `POST {prefix}/keymap/reset` — same error shapes as `putKeymap`. */
export async function resetKeymap(
  body: KeymapResetRequest,
  signal?: AbortSignal,
): Promise<KeymapPutResponse> {
  return apiFetch<KeymapPutResponse>(
    `${scoped()}/keymap/reset`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

export interface KeymapRevisionConflictDetail {
  message: string;
  current_revision: number;
}

export function keymapRevisionConflictDetail(
  e: unknown,
): KeymapRevisionConflictDetail | null {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (d.error !== 'revision_conflict') return null;
  if (typeof d.current_revision !== 'number') return null;
  return {
    message: typeof d.message === 'string' ? d.message : 'The keymap changed elsewhere.',
    current_revision: d.current_revision,
  };
}

export interface KeymapClassConflictDetail {
  message: string;
  current_revision: number;
  report: KeymapValidationReport;
  class_conflicts: KeymapClassConflict[];
}

export function keymapClassConflictDetail(e: unknown): KeymapClassConflictDetail | null {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (d.error !== 'class_hotkey_conflict') return null;
  return {
    message: typeof d.message === 'string' ? d.message : 'That key is bound to a class.',
    current_revision: typeof d.current_revision === 'number' ? d.current_revision : 0,
    report: (d.report as KeymapValidationReport) ?? {
      ok: false,
      errors: [],
      warnings: [],
      force_allowed: false,
    },
    class_conflicts: Array.isArray(d.class_conflicts)
      ? (d.class_conflicts as KeymapClassConflict[])
      : [],
  };
}

export interface KeymapValidationFailedDetail {
  message: string;
  current_revision: number;
  report: KeymapValidationReport;
}

export function keymapValidationFailedDetail(
  e: unknown,
): KeymapValidationFailedDetail | null {
  if (!(e instanceof ApiError) || e.status !== 422) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (d.error !== 'validation_failed') return null;
  return {
    message: typeof d.message === 'string' ? d.message : 'The keymap has an error.',
    current_revision: typeof d.current_revision === 'number' ? d.current_revision : 0,
    report: (d.report as KeymapValidationReport) ?? {
      ok: false,
      errors: [],
      warnings: [],
      force_allowed: false,
    },
  };
}
