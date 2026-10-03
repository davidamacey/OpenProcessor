/**
 * Typed API client for the OpenProcessor curation endpoints, mounted
 * under `API_PREFIX` (`/curation`).
 *
 * - Single base URL, defaulting to `''` (empty → relative paths, proxied
 *   by nginx in Docker production).
 * - `apiFetch` retries on 5xx with exponential backoff (3 tries, 250 / 500 / 1000ms).
 * - A 503's `Retry-After` header (seconds) replaces that attempt's fixed
 *   delay, clamped to `MAX_RETRY_AFTER_MS` — never an extra attempt or an
 *   unbounded wait (OpenProcessor 3cd4ca87's Triton-unavailable 503s send
 *   `Retry-After: 5`).
 * - Caller-supplied AbortSignal is honoured; cancellation never retries.
 * - On 4xx the original ApiError is thrown immediately (no retry).
 *
 * All endpoint URL patterns come from Section "Phase 2D" of the v7 plan.
 */

import { parseMethodsResponse, type MethodsResponse } from './strategies';
import { parseCurationSettings, type CurationSettings } from '$lib/curationSettings';
import { mapCropSlots } from './annotations/cropSlots';
import type { KeymapDocument, KeymapValidationIssue } from './keymapFallback';
import type { XYXY, SlotKey, SlotData, SlotSpec } from './annotations/types';
import type { DatasetExportSpec } from './annotations/datasetExport';
import type { RegionBoxInput } from './annotations/multiBox';
import {
  isNoRegionProfileDetail,
  notifyRegionProfileUnavailable,
  REGION_PROFILE_UNAVAILABLE_MESSAGE,
} from './regionProfileUnavailable';
import type {
  BulkLabelConflict,
  BulkLabelResult,
  ClassesResponse,
  ClassMergeDryRun,
  ClassThresholds,
  ClusterFilter,
  CropFilter,
  CropHistoryResponse,
  CropContextResponse,
  CropRegionUndoBatchResult,
  CropUndoBatchResult,
  ItemTextLine,
  RegistryClassCreate,
  RegistryClassMerge,
  RegistryClassUpdate,
  ResolveNewClassRequest,
  ResolveNewClassResponse,
  Cluster,
  Crop,
  ExportDatasetList,
  ExportResult,
  ExportStatus,
  ApiHealth,
  GlobalHealth,
  ServedRegionProfile,
  SingleClassExportResult,
  SingleClassExportStatus,
  ModelsStatus,
  StatsSummary,
  TestHoldoutFreezeResult,
  TestHoldoutStats,
  DiverseSelection,
  PaginatedResponse,
  ReviewItem,
  ReviewTab,
  SearchCrop,
  SelectDiverseScope,
  SelectJobStatus,
  UnloadModelResponse,
  IngestStatus,
  RegionDrain,
  IngestPathLookupResponse,
  IngestBatchRequest,
  IngestUploadRequest,
  BatchIngestResponse,
  IngestConfig,
} from './types';
import type {
  AugmentationPresetsResponse,
  CancelResponse,
  LogTailResponse,
  PresetsResponse,
  PreflightReport,
  ProfilesResponse,
  PromoteRequest,
  PromoteResponse,
  RunsListResponse,
  StartCampaignResponse,
  StartTrainResponse,
  TrainCampaignSpec,
  TrainJobSpec,
  TrainJobStatus,
  TrainManifest,
} from './types_train';
import type {
  DatasetErrorDetail,
  DatasetFormatsResponse,
  DatasetImportEntryPage,
  DatasetImportJob,
  DatasetImportList,
  DatasetImportRequest,
  DatasetIssuePage,
  DatasetPreview,
  DatasetPreviewRequest,
  DatasetUndoReport,
  DatasetUndoRequest,
  DatasetUploadResponse,
  NextStep,
  ReprocessJob,
  ReprocessOneRequest,
  ReprocessRequest,
  ReprocessResponse,
} from './types_import';
import type {
  ActivateResponse,
  ActiveConfigResponse,
  ActiveRef,
  ConfigActivateRequest,
  ConfigCloneRequest,
  ConfigErrorDetail,
  ConfigRevisionList,
  ValidationReport,
} from './types_config';
import type {
  PackUpdateRequest,
  PackValidateRequest,
  PromptPackDoc,
  PromptPackList,
  PromptPackSchema,
} from './types_packs';
import type {
  ActivationImpact,
  ConfigVocabulary,
  ProfileActivateResponse,
  ProfileUpdateRequest,
  ProfileValidateRequest,
  RegionProfileDoc,
  RegionProfileList,
  RegionProfileSchema,
} from './types_profiles';
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
} from './types_bakeoff';
import type {
  ArchiveRequest,
  CloneSettingsRequest,
  CreateProjectRequest,
  DeleteDryRunResponse,
  PatchProjectRequest,
  PipelinePauseState,
  ProjectErrorDetail,
  ProjectLifecycleResponse,
  ProjectRecordResponse,
  ProjectsResponse,
} from './types_projects';
import type {
  ModelClassMappingResponse,
  ModelSharingRequest,
  ModelSharingResponse,
} from './types_models';

// Vite exposes only PUBLIC_-prefixed env vars to the client. SvelteKit uses
// `$env/dynamic/public` but importing that here would force every consumer
// onto SSR-only paths; we go through `import.meta.env` so this module remains
// usable in pure client contexts (and, for production, the value is baked in
// at build time and overridable via the docker-entrypoint shim).
// Empty default: in Docker the labeler's nginx proxies {API_PREFIX}/* to the
// backend on the same docker network. Relative URLs work from any LAN IP / VPN client.
// For local `npm run dev` outside Docker, set PUBLIC_TRITON_API_URL=http://localhost:4603 in .env.
const RAW_BASE = (import.meta.env?.PUBLIC_TRITON_API_URL as string | undefined) ?? '';

export const apiBase: string = RAW_BASE.replace(/\/+$/, '');

/**
 * Path prefix every backend endpoint hangs off, e.g. `/curation/health`.
 * Defaults to OpenProcessor's `OP_API_PREFIX` default, `/curation`, and
 * must equal it: the backend also builds some URLs itself (thumbnail
 * URLs in `/regions` rows) from its own prefix, and nginx only proxies
 * this one. Flipped from a transitional prefix at T-E2
 * (`docs/design/backend-integration-phase-b-plan-2026-09-20.md`).
 */
const RAW_API_PREFIX = (import.meta.env?.PUBLIC_API_PREFIX as string | undefined) ?? '';

/**
 * Empty and `__API_PREFIX__` both mean "unset", on purpose.
 *
 * The production image bakes the literal `__API_PREFIX__` at build time
 * and `sed`s it at container start (`docker-entrypoint.sh`). If the env
 * var is unset there the substitution yields `''`, and `?? '/curation'` would
 * NOT fire — `??` only catches null/undefined — producing prefix-less
 * URLs like `/health`. If the entrypoint is skipped entirely the
 * placeholder leaks through verbatim. Both are total, silent failures;
 * both are cheaper to absorb here than to debug in a container.
 */
export function normalizeApiPrefix(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed || trimmed.startsWith('__')) return '/curation';
  const leading = trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
  return leading.replace(/\/+$/, '');
}

export const API_PREFIX: string = normalizeApiPrefix(RAW_API_PREFIX);

/**
 * Multi-project scoping (P1, `docs/design/
 * any-domain-rev3-and-projects-contract-review-2026-09-26.md` §7;
 * OWNER DECISION: no backward compatibility with the retired unscoped
 * `{API_PREFIX}/...` alias). Every route except the GLOBAL ones below
 * lives under a project's own served `prefix`
 * (`/curation/projects/{project}/...`, from `GET {globalApi()}/projects`).
 * There is no `default` fallback prefix baked in here — the active
 * project is set by `setScopedPrefix()` once `projectsStore.load()`
 * resolves the default project, and every scoped call made before that
 * throws (fails closed, matching the backend's `ProjectNotBound`).
 */
const scopeHolder: { prefix: string | null; generation: number } = {
  prefix: null,
  generation: 0,
};

export class ProjectNotSelectedError extends Error {
  constructor() {
    super('no active project selected yet');
    this.name = 'ProjectNotSelectedError';
  }
}

/**
 * Sets the active project's scoped prefix (e.g. `/curation/projects/
 * default`), as served by `GET {globalApi()}/projects`. Called once by
 * `projectsStore.load()` at boot; never persisted (no localStorage) —
 * the active project is always live UI state, seeded fresh from the
 * served project list every load.
 */
export function setScopedPrefix(prefix: string): void {
  if (scopeHolder.prefix === prefix) return;
  scopeHolder.prefix = prefix;
  scopeHolder.generation += 1;
}

/** Bumped every time the active project's scoped prefix changes — the
 *  stale-response guard in `apiFetch` compares a request's start
 *  generation against the current one. */
export function scopeGeneration(): number {
  return scopeHolder.generation;
}

/**
 * A scoped response that arrives after the active project changed. It
 * is an `AbortError` (every call site already treats an abort as "drop
 * it silently"), so a late answer from the previous project never
 * renders in the new one.
 */
export function staleProjectError(): DOMException {
  return new DOMException('response from a previous project', 'AbortError');
}

/**
 * The one function every scoped backend call builds its URL through,
 * e.g. `` `${scoped()}/health` ``. Throws `ProjectNotSelectedError` if
 * no project has been selected yet — every scoped call site should only
 * ever run after the root layout's project bootstrap has resolved.
 * Distinct from `globalApi()` below for the small set of routes that
 * are never project-scoped (`/projects`, the global `/health`/`/events`).
 */
export function scoped(): string {
  if (scopeHolder.prefix === null) throw new ProjectNotSelectedError();
  return scopeHolder.prefix;
}

/**
 * Key for client-side caches that must never bleed data across
 * projects (crop-id-keyed caches, the undo ring buffer) — crop ids are
 * content-derived, so the same image gets the same `crop_id` in every
 * project. Always mirrors `scoped()`'s current value: the prefix
 * already uniquely identifies the active project, so there's no reason
 * for a second, independently-settable holder.
 */
export function activeProjectKey(): string {
  return scoped();
}

/**
 * Builder for the handful of routes that stay global (never
 * project-scoped): `/projects` (list/CRUD), the global `/health` and
 * the global `/events` stream. No call site here builds a scoped URL
 * from this — it is always `API_PREFIX` itself.
 */
export function globalApi(): string {
  return API_PREFIX;
}

const DETAIL_MAX_CHARS = 200;

/**
 * Pull the human-readable reason out of an error response body.
 *
 * The backend is FastAPI, so 4xx bodies are `{detail: "..."}` (occasionally
 * `{message: "..."}`, or a plain-text body). Without this, every toast in
 * the app shows `API 422 http://…/batch_label` and the operator has no idea
 * what the server objected to — the callsites all render `Error.message`.
 */
/**
 * p1 (2026-09-24 interactive pass): humanize one pydantic/FastAPI
 * validation-error entry (`{loc: ['body', 'name'], msg: "String should
 * match pattern '^[a-z0-9_]+$'", ...}`) into "name: must match pattern
 * ^[a-z0-9_]+$" — the field it's actually complaining about, and the
 * one most common message shape reworded away from pydantic's internal
 * phrasing. Still entirely the server's own field name / regex / value;
 * this never invents validation rules of its own.
 */
export function formatValidationEntry(entry: unknown): string | null {
  if (!entry || typeof entry !== 'object') return null;
  const e = entry as { loc?: unknown; msg?: unknown };
  if (typeof e.msg !== 'string' || !e.msg) return null;
  let msg = e.msg;
  const patternMatch = /^String should match pattern '(.+)'$/.exec(msg);
  if (patternMatch) msg = `must match pattern ${patternMatch[1]}`;

  const loc = Array.isArray(e.loc) ? e.loc : [];
  const field = loc
    .filter((p): p is string => typeof p === 'string' && p !== 'body' && p !== 'query')
    .pop();
  return field ? `${field}: ${msg}` : msg;
}

function errorDetail(body: unknown): string | null {
  let raw: unknown = null;
  if (typeof body === 'string') {
    raw = body;
  } else if (body && typeof body === 'object') {
    const rec = body as Record<string, unknown>;
    raw = rec.detail ?? rec.message ?? null;
    // Pydantic/FastAPI validation errors (`{detail: [{loc, msg, ...}, ...]}`,
    // e.g. the 422 for a class name that doesn't match `^[a-z0-9_]+$`) —
    // join every entry's `msg` so the server's validation text reaches the
    // toast instead of falling through to a generic "API 422" message.
    // p1 (2026-09-24 interactive pass): the raw pydantic sentence
    // ("String should match pattern '^[a-z0-9_]+$'") gave no field
    // context and read as internal-error jargon. Each entry now renders
    // "field: message" (from the server's own `loc`), and the one most
    // common pydantic phrasing gets a lighter wording — still the
    // server's own regex/value, never invented.
    if (Array.isArray(raw)) {
      const msgs = raw
        .map((entry) => formatValidationEntry(entry))
        .filter((m): m is string => !!m);
      raw = msgs.length ? msgs.join('; ') : null;
    }
    // Structured FastAPI details (`{detail: {error, ...}}`) carry their
    // human-readable text under `error`.
    if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
      raw = (raw as Record<string, unknown>).error ?? null;
    }
  }
  if (typeof raw !== 'string') return null;
  const text = raw.trim();
  if (!text) return null;
  return text.length > DETAIL_MAX_CHARS
    ? `${text.slice(0, DETAIL_MAX_CHARS - 1)}…`
    : text;
}

/** The 422 `detail` an unknown per-run strategy id (e.g. `prompt_pack`)
 *  produces on `auto_label/start`. */
export interface UnknownStrategyDetail {
  axis: string;
  requested: string;
  valid_ids: string[];
}

export function unknownStrategyDetail(e: unknown): UnknownStrategyDetail | null {
  if (!(e instanceof ApiError) || e.status !== 422) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (typeof d.axis !== 'string' || typeof d.requested !== 'string') return null;
  if (!Array.isArray(d.valid_ids)) return null;
  return {
    axis: d.axis,
    requested: d.requested,
    valid_ids: d.valid_ids.filter((v): v is string => typeof v === 'string'),
  };
}

export class ApiError extends Error {
  status: number;
  body: unknown;
  url: string;
  /** The server's `detail`/`message` string, when it sent one. */
  detail: string | null;
  constructor(status: number, url: string, body: unknown, message?: string) {
    const detail = errorDetail(body);
    super(message ?? `API ${status} ${url}${detail ? ` — ${detail}` : ''}`);
    this.status = status;
    this.body = body;
    this.url = url;
    this.detail = detail;
  }
}

/** A region route's 409 when the backend has no region profile. See
 *  `./regionProfileUnavailable.ts` for how the UI absorbs it. */
export class RegionProfileUnavailableError extends ApiError {
  constructor(url: string, body: unknown) {
    super(409, url, body, REGION_PROFILE_UNAVAILABLE_MESSAGE);
    this.name = 'RegionProfileUnavailableError';
  }
}

const RETRY_DELAYS_MS = [250, 500, 1000];

/**
 * Cap on how long a single 503 `Retry-After` honour can push a retry's
 * wait out to — the backend's Triton-unavailable handler sends `5`
 * (`TRITON_UNAVAILABLE_RETRY_AFTER_SECONDS`, OpenProcessor 3cd4ca87), but
 * this bounds any served value so a misbehaving/huge header can't stall
 * the UI far past the existing retry budget. It replaces one attempt's
 * fixed backoff delay — it never adds an attempt.
 */
const MAX_RETRY_AFTER_MS = 5000;

/**
 * `Retry-After` on a 503, in ms — seconds only (the only form any
 * OpenProcessor 503 sends today), clamped to a sane non-negative range.
 * Null when absent/unparseable, so the caller falls back to the normal
 * fixed backoff delay for that attempt.
 */
function parseRetryAfterMs(res: Response): number | null {
  const header = res.headers.get('Retry-After');
  if (!header) return null;
  const secs = Number(header);
  if (!Number.isFinite(secs) || secs < 0) return null;
  return secs * 1000;
}

async function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const id = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(id);
        reject(new DOMException('Aborted', 'AbortError'));
      },
      { once: true },
    );
  });
}

/**
 * Resolve a path/URL against the configured `apiBase`. Idempotent — an
 * already-absolute URL (or one already prefixed with `apiBase`) passes
 * through unchanged, so it's safe to call on a value that might have
 * already been resolved upstream (e.g. a server-supplied
 * `representative_thumb_urls` entry rendered through a shared helper).
 *
 * The backend intentionally emits relative `{API_PREFIX}/...` URLs in API
 * response bodies (e.g. `region_thumbnail_url`, `representative_thumb_urls`)
 * so the same payload works both same-origin (production nginx proxy,
 * empty `apiBase`) and cross-origin (a remote `PUBLIC_TRITON_API_URL`).
 * Resolving those against `apiBase` is the client's job — every render
 * site that puts a server-supplied URL into an `<img src>` must go
 * through this function first.
 */
export function resolveApiUrl(url: string): string {
  if (url.startsWith('http')) return url;
  if (apiBase && url.startsWith(apiBase)) return url;
  return `${apiBase}${url}`;
}

export interface ApiFetchOptions {
  /** A GLOBAL route (`/projects*`, global `/health`): never dropped as
   *  stale when the active project changes mid-request. */
  global?: boolean;
}

export async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
  opts: ApiFetchOptions = {},
): Promise<T> {
  const url = resolveApiUrl(path);
  // Stale-project guard (review §7.1): a scoped request remembers the
  // project it was built for; if the active project changed before its
  // response lands, the response is dropped as an AbortError.
  const startPrefix = scopeHolder.prefix;
  const startGeneration = scopeHolder.generation;
  const isScopedCall =
    !opts.global && startPrefix !== null && path.startsWith(`${startPrefix}/`);
  const assertFresh = (): void => {
    if (isScopedCall && scopeHolder.generation !== startGeneration) {
      throw staleProjectError();
    }
  };
  let attempt = 0;
  let lastError: unknown;
  // 1 initial + 3 retries on 5xx => 4 attempts max.
  for (; attempt < RETRY_DELAYS_MS.length + 1; attempt++) {
    let retryAfterMs: number | null = null;
    try {
      const res = await fetch(url, {
        ...init,
        signal: signal ?? init.signal ?? null,
        headers: {
          Accept: 'application/json',
          // A multipart body (ingestUpload) must let the browser set its
          // own `Content-Type: multipart/form-data; boundary=...` — a
          // hardcoded `application/json` here breaks the boundary and the
          // backend can't parse the parts at all.
          ...(init.body && !(init.body instanceof FormData)
            ? { 'Content-Type': 'application/json' }
            : {}),
          ...(init.headers ?? {}),
        },
      });
      assertFresh();
      if (res.ok) {
        if (res.status === 204) return undefined as T;
        const ct = res.headers.get('content-type') ?? '';
        const out = ct.includes('application/json')
          ? ((await res.json()) as T)
          : ((await res.text()) as unknown as T);
        assertFresh();
        return out;
      }
      let body: unknown = null;
      try {
        body = await res.json();
      } catch {
        try {
          body = await res.text();
        } catch {
          /* ignore */
        }
      }
      if (res.status === 409 && isNoRegionProfileDetail(errorDetail(body))) {
        notifyRegionProfileUnavailable();
        throw new RegionProfileUnavailableError(url, body);
      }
      const err = new ApiError(res.status, url, body);
      // Don't retry on 4xx — they won't get better.
      if (res.status < 500) throw err;
      if (res.status === 503) retryAfterMs = parseRetryAfterMs(res);
      lastError = err;
    } catch (e) {
      if (e instanceof ApiError && e.status < 500) throw e;
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      lastError = e;
    }
    // A retry after the project changed would fetch the OLD project's
    // URL again — stop instead.
    assertFresh();
    if (attempt < RETRY_DELAYS_MS.length) {
      // A served `Retry-After` (503 only) replaces this attempt's fixed
      // backoff delay, clamped to MAX_RETRY_AFTER_MS — it never adds an
      // attempt or extends the total retry budget.
      const delay =
        retryAfterMs != null
          ? Math.min(retryAfterMs, MAX_RETRY_AFTER_MS)
          : RETRY_DELAYS_MS[attempt]!;
      // An abort during the backoff propagates — the caller cancelled.
      await sleep(delay, signal);
    }
  }
  throw lastError ?? new Error(`apiFetch failed: ${url}`);
}

// -- query string helpers ------------------------------------------------

/** Builds `?k=v&...`, skipping null/undefined. An array value is sent as
 *  one `k=v` per element (FastAPI list query params read repeated keys, not
 *  a comma-joined string); an empty array is omitted. Exported for tests. */
export function qs(params: Record<string, unknown>): string {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null) continue;
    if (Array.isArray(v)) {
      for (const x of v) u.append(k, String(x));
      continue;
    }
    u.set(k, String(v));
  }
  const s = u.toString();
  return s ? `?${s}` : '';
}

// -- endpoints -----------------------------------------------------------

export function getHealth(signal?: AbortSignal): Promise<ApiHealth> {
  return apiFetch<ApiHealth>(`${scoped()}/health`, {}, signal);
}

/** `GET {globalApi()}/health` — unscoped, no project bound. Feeds only
 *  the top-bar API status chip. */
export function getGlobalHealth(signal?: AbortSignal): Promise<GlobalHealth> {
  return apiFetch<GlobalHealth>(`${globalApi()}/health`, {}, signal, { global: true });
}

/** `GET {globalApi()}/projects` — the switcher vocabulary, global
 *  (unscoped). The one read every project-scoped call depends on: a
 *  project's `prefix` here is what `setScopedPrefix()` is seeded with.
 *  `includeArchived` adds the served `archived` projects (the list
 *  membership per status is the server's, never filtered here). */
export function getProjects(
  signal?: AbortSignal,
  includeArchived = false,
): Promise<ProjectsResponse> {
  return apiFetch<ProjectsResponse>(
    `${globalApi()}/projects${qs({ include_archived: includeArchived ? true : undefined })}`,
    {},
    signal,
    { global: true },
  );
}

// -- project lifecycle (P3; all GLOBAL, never scoped) ---------------------

/** `GET {globalApi()}/projects/{slug}` — the record for a slug the
 *  default list doesn't carry (an archived project's deep link). */
export function getProject(
  slug: string,
  signal?: AbortSignal,
): Promise<ProjectRecordResponse> {
  return apiFetch<ProjectRecordResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}`,
    {},
    signal,
    { global: true },
  );
}

export function createProject(
  body: CreateProjectRequest,
): Promise<ProjectLifecycleResponse> {
  return apiFetch<ProjectLifecycleResponse>(
    `${globalApi()}/projects`,
    { method: 'POST', body: JSON.stringify(body) },
    undefined,
    { global: true },
  );
}

export function patchProject(
  slug: string,
  body: PatchProjectRequest,
): Promise<ProjectLifecycleResponse> {
  return apiFetch<ProjectLifecycleResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}`,
    { method: 'PATCH', body: JSON.stringify(body) },
    undefined,
    { global: true },
  );
}

export function archiveProject(
  slug: string,
  body: ArchiveRequest,
): Promise<ProjectLifecycleResponse> {
  return apiFetch<ProjectLifecycleResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}/archive`,
    { method: 'POST', body: JSON.stringify(body) },
    undefined,
    { global: true },
  );
}

export function unarchiveProject(
  slug: string,
  body: ArchiveRequest,
): Promise<ProjectLifecycleResponse> {
  return apiFetch<ProjectLifecycleResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}/unarchive`,
    { method: 'POST', body: JSON.stringify(body) },
    undefined,
    { global: true },
  );
}

export function cloneProjectSettings(
  slug: string,
  body: CloneSettingsRequest,
): Promise<ProjectLifecycleResponse> {
  return apiFetch<ProjectLifecycleResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}/clone_settings`,
    { method: 'POST', body: JSON.stringify(body) },
    undefined,
    { global: true },
  );
}

/** `DELETE …?dry_run=true` — the served report, writes nothing. */
export function deleteProjectDryRun(slug: string): Promise<DeleteDryRunResponse> {
  return apiFetch<DeleteDryRunResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}${qs({ dry_run: true })}`,
    { method: 'DELETE' },
    undefined,
    { global: true },
  );
}

/** `DELETE …?confirm=<slug>` — a real, guarded delete. Answers 202 with
 *  the `deleting` record; the removal finishes in the background. */
export function deleteProject(
  slug: string,
  confirm: string,
): Promise<ProjectLifecycleResponse> {
  return apiFetch<ProjectLifecycleResponse>(
    `${globalApi()}/projects/${encodeURIComponent(slug)}${qs({ confirm })}`,
    { method: 'DELETE' },
    undefined,
    { global: true },
  );
}

/**
 * The structured `{detail: {error, message, ...}}` body every project
 * route answers an error with (`ConfigErrorDetail`), or `null` when the
 * error isn't one (a network failure, a plain-string detail, a pydantic
 * validation list). The UI renders `message` verbatim and branches only
 * on the served `error` code.
 */
export function projectErrorDetail(e: unknown): ProjectErrorDetail | null {
  if (!(e instanceof ApiError)) return null;
  const body = e.body;
  if (!body || typeof body !== 'object') return null;
  const detail = (body as { detail?: unknown }).detail;
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return null;
  const d = detail as Record<string, unknown>;
  if (typeof d.error !== 'string' || typeof d.message !== 'string') return null;
  return d as unknown as ProjectErrorDetail;
}

/** The text to show for a failed project action: the served `message`
 *  when the error is structured, else the generic `ApiError.detail`
 *  (e.g. a joined pydantic validation list), else the error's message. */
export function projectErrorText(e: unknown): string {
  const d = projectErrorDetail(e);
  if (d) return d.message;
  if (e instanceof ApiError && e.detail) return e.detail;
  return (e as Error)?.message ?? String(e);
}

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
// backend that predates OpenProcessor W2b 404s/501s `GET {prefix}/keymap`
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

/** `PUT /classes/{id}` 422 `hotkey_reserved` (plan §4.5). */
export interface HotkeyReservedDetail {
  message: string;
  actions: Array<{ action_id: string; context: string; label: string }>;
}

export function hotkeyReservedDetail(e: unknown): HotkeyReservedDetail | null {
  if (!(e instanceof ApiError) || e.status !== 422) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (d.error !== 'hotkey_reserved') return null;
  return {
    message: typeof d.message === 'string' ? d.message : 'That letter is reserved.',
    actions: Array.isArray(d.actions)
      ? (d.actions as HotkeyReservedDetail['actions'])
      : [],
  };
}

/** `PUT /classes/{id}` 409 `hotkey_taken` (plan §4.5). */
export interface HotkeyTakenDetail {
  message: string;
  class_id: number;
  class_name: string;
}

export function hotkeyTakenDetail(e: unknown): HotkeyTakenDetail | null {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (d.error !== 'hotkey_taken') return null;
  if (typeof d.class_id !== 'number' || typeof d.class_name !== 'string') return null;
  return {
    message:
      typeof d.message === 'string' ? d.message : `Already bound to ${d.class_name}.`,
    class_id: d.class_id,
    class_name: d.class_name,
  };
}

// -- embedding projection (2-d visualization overlay, Phase 5) -----------
//
// docs/curation-strategy-plan-2026-09.md §2.7/§5.6/§7 — `embedding_viz.py`
// + `viz.py` (OpenProcessor). UMAP-as-a-visualization-only overlay is the
// one method in the whole curation-strategy plan that was NOT validated
// in Phase 2 before implementation started; its own §6 acceptance bar
// (2-d neighborhood purity vs. the real IVF cluster_id) decides whether
// it ships plain, ships behind an "approximate" banner, or doesn't ship
// at all. Nothing here assumes an outcome — `isEmbeddingVizAvailable`
// (strategies.ts) is what actually gates whether any UI renders at all,
// exactly like `isDiverseOverlayAvailable` gates Phase 4's `diverse`
// overlay.
//
// Coordinates are cached/batch-computed server-side (plan §2.7's
// non-negotiable rule: "never fit on a request path") — `getVizProjection`
// only ever serves whatever the last `rebuildVizProjection()` job
// produced.

/**
 * One projected point. Color is derived client-side from `cluster_id`
 * (see `colorForCluster` in `embeddingPlot.ts`) — this overlay decorates
 * an existing assignment, it never computes or chooses one.
 */
export interface VizPoint {
  crop_id: string;
  x: number;
  y: number;
  cluster_id: number | null;
  class_name: string | null;
  class_source: string | null;
}

export interface VizProjectionResponse {
  points: VizPoint[];
  /** `points.length`, kept as a field for parity with other list responses
   *  in this file — the server doesn't send a separate total, there's no
   *  pagination here (`max_points` is a hard cap, not a page size). */
  total: number;
  /**
   * CONFIRMED (2026-09-10) against the real `GET {API_PREFIX}/viz/projection`
   * (`embedding_viz.get_cached_projection`): the server returns
   * `{status: 'not_built'}` when nothing has been fit yet, or
   * `{points, projection_version, fitted_at, stale}` otherwise — there is
   * no `built`/`not_built` boolean on the wire. `built` here is this
   * file's own derived convenience (`status !== 'not_built'`), kept so
   * `EmbeddingPlot` doesn't need to know the raw sentinel shape.
   */
  built: boolean;
  fitted_at: string | null;
  projection_version: string | null;
  /** True iff some in-scope crop's cached coordinates predate the latest
   *  fit (partial-coverage signal, same philosophy as the scores
   *  endpoints' `field_coverage`) — `false` when `built` is `false`. */
  stale: boolean;
}

const EMPTY_VIZ_PROJECTION: VizProjectionResponse = {
  points: [],
  total: 0,
  built: false,
  fitted_at: null,
  projection_version: null,
  stale: false,
};

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

function parseVizPoint(raw: unknown): VizPoint | null {
  if (!isPlainObject(raw)) return null;
  const { crop_id, x, y } = raw;
  if (typeof crop_id !== 'string' || !crop_id) return null;
  if (typeof x !== 'number' || !Number.isFinite(x)) return null;
  if (typeof y !== 'number' || !Number.isFinite(y)) return null;
  const cluster_id =
    typeof raw.cluster_id === 'number' && Number.isFinite(raw.cluster_id)
      ? raw.cluster_id
      : null;
  return {
    crop_id,
    x,
    y,
    cluster_id,
    class_name: typeof raw.class_name === 'string' ? raw.class_name : null,
    class_source: typeof raw.class_source === 'string' ? raw.class_source : null,
  };
}

/**
 * Fetch the cached 2-d projection. **Never rejects** (mirrors
 * `getMethods`'s contract) — `{API_PREFIX}/viz/projection` may not exist yet (the
 * backend Phase 5 lands independently of this frontend branch) or may
 * 404/5xx for any other reason, and a fetch failure here should degrade
 * `EmbeddingPlot` to its pending/empty state rather than crash the page
 * it replaced the grid on. A caller-initiated abort still propagates —
 * that's a cancellation, not a backend failure.
 *
 * The real payload is `{status: 'not_built'}` (nothing fit yet) or
 * `{points, projection_version, fitted_at, stale}` (confirmed 2026-09-10
 * against `embedding_viz.get_cached_projection`) — normalized here into
 * this file's own `built`/`fitted_at`/`projection_version`/`stale` shape.
 */
export async function getVizProjection(
  params: {
    cluster_id?: number | null;
    class_id?: number | null;
    max_points?: number;
  } = {},
  signal?: AbortSignal,
): Promise<VizProjectionResponse> {
  try {
    const raw = await apiFetch<unknown>(
      `${scoped()}/viz/projection${qs({
        cluster_id: params.cluster_id ?? undefined,
        class_id: params.class_id ?? undefined,
        max_points: params.max_points ?? undefined,
      })}`,
      {},
      signal,
    );
    if (!isPlainObject(raw)) return EMPTY_VIZ_PROJECTION;
    if (raw.status === 'not_built') return EMPTY_VIZ_PROJECTION;
    const rawPoints = Array.isArray(raw.points) ? raw.points : [];
    const points = rawPoints.map(parseVizPoint).filter((p): p is VizPoint => p != null);
    return {
      points,
      total: points.length,
      built: true,
      fitted_at: typeof raw.fitted_at === 'string' ? raw.fitted_at : null,
      projection_version:
        typeof raw.projection_version === 'string' ? raw.projection_version : null,
      stale: raw.stale === true,
    };
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    return EMPTY_VIZ_PROJECTION;
  }
}

/**
 * Background rebuild-job snapshot. CONFIRMED (2026-09-10) against the
 * real `embedding_viz._JobState` — flat, not the nested
 * `{running, result: {...}}` shape this file originally guessed:
 * `status` is a string enum (`'idle' | 'running' | 'completed' |
 * 'failed' | 'cancelled'`), timestamps are unix-epoch numbers (`0` when
 * unset, not `null`), and `n_written`/`projection_version` are top-level
 * fields, not nested under a `result` key.
 */
export interface VizProjectionJob {
  job_id: string;
  status: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';
  scope: string;
  cluster_id: number | null;
  n_pool: number;
  n_written: number;
  started_at: number;
  finished_at: number;
  error: string | null;
  projection_version: string | null;
}

/**
 * Kick off a projection (re)fit — a background job, not a request-path
 * fit (plan §2.7/§3: "coordinates cached/batch-computed... never fit on
 * a request path"). Unlike `getVizProjection`, this is an explicit
 * user-initiated action (the operator clicked "Rebuild"), so — matching
 * the `buildRegionFpCentroids`/`clusterRegions` precedent — it lets the
 * error propagate for the caller to catch + toast rather than swallowing
 * it into a fallback value.
 */
export function rebuildVizProjection(signal?: AbortSignal): Promise<VizProjectionJob> {
  return apiFetch<VizProjectionJob>(
    `${scoped()}/viz/projection/rebuild`,
    { method: 'POST' },
    signal,
  );
}

/** Current rebuild-job snapshot — poll this after `rebuildVizProjection()`. */
export function getVizProjectionStatus(signal?: AbortSignal): Promise<VizProjectionJob> {
  return apiFetch<VizProjectionJob>(`${scoped()}/viz/projection/status`, {}, signal);
}

/** Cancel a running rebuild. `cancelled` is false when nothing was running. */
export function cancelVizProjection(
  signal?: AbortSignal,
): Promise<VizProjectionJob & { cancelled: boolean }> {
  return apiFetch<VizProjectionJob & { cancelled: boolean }>(
    `${scoped()}/viz/projection/cancel`,
    { method: 'POST' },
    signal,
  );
}

// -- region browse / training-cohort selection --------------------------

/**
 * Base path for the region/annotation-slot collection endpoints (the
 * region browse sub-system below — cluster, FP centroids, training
 * candidates, etc). The single place this path lives: every call site
 * below builds on it, and `regionRouteScan.test.ts` fails the build if a
 * bare path literal reappears.
 */
const REGION_BASE = '/regions';

/** One row of a region browse route (`RegionRow`): the full item plus the
 *  box the row is about. Every per-box value (geometry, score, detector,
 *  text, cluster) is an element of `region_boxes`, read through
 *  `slots[slotKey].subBoxes`; the row's own box is the one whose id equals
 *  `region_box_id`. */
export interface RegionBrowseItem {
  crop_id: string;
  id: string;
  image_path: string;
  /** The source image's id; targets an image Reprocess. */
  image_id?: string;
  bbox_norm: number[];
  region_status: string | null;
  region_verified: boolean | null;
  region_validated: boolean | null;
  region_detector_chain: string[] | null;
  region_detected_at: string | null;
  region_verifier: string | null;
  region_verifier_version: string | null;
  region_verified_at: string | null;
  region_rejection_reason: string | null;
  region_visible: boolean | null;
  class_id: number | null;
  class_name: string | null;
  cluster_id: number | null;
  /** Parent-crop rank by size in its image (1 = largest). */
  crop_rank_in_image?: number | null;
  crop_area_norm?: number | null;
  updated_at: string;
  thumbnail_url?: string;
  selection_reason?: string;
  /** Per-slot capability data — see `Crop.slots` in types.ts. Added by
   *  `getRegions` via `mapCropSlots`. */
  slots?: Record<SlotKey, SlotData>;
  /** The box this row is about; null for an item-level row. */
  region_box_id?: string | null;
}

export interface RegionsPage {
  total: number;
  page: number;
  page_size: number;
  items: RegionBrowseItem[];
  mode?: string;
  selection_reason?: string;
  /** `RegionRowPage.total_rows` — counts ROWS (boxes, when the request
   *  selects boxes), vs. `total` which counts ITEMS (the unit pages
   *  paginate). Shown beside the item count when they differ — see
   *  `SlotGallery.svelte`'s count chip. */
  total_rows?: number;
  /** True when an item on this page matched more boxes than the index
   *  reports per item, so some of its rows are missing (`total_rows`
   *  still counts them). */
  rows_truncated?: boolean;
}

export interface RegionsQuery {
  page?: number;
  page_size?: number;
  class_id?: number;
  cluster_id?: number;
  /** Region clustering bucket (independent of the item cluster_id). */
  region_cluster_id?: number;
  /** AHC region sub-cluster id (e.g. "17a"). */
  region_cluster_subid?: string;
  /** Order a bucket's regions by sub-cluster so AHC groups come back contiguous. */
  sort_by_subid?: boolean;
  /** Only regions on the top-N largest crops (crop_rank_in_image<=N). */
  max_rank?: number;
  min_score?: number;
  max_score?: number;
  verified?: boolean;
  detector?: string;
  text?: string;
  include_test?: boolean;
  /** Region lifecycle status (`GET {API_PREFIX}/regions/statuses` for the
   *  deployment's vocabulary) — the backend 400s on an unknown value.
   *  Independent of `verified`, which is a boolean, not a status. */
  status?: string;
  /** Per-box state filter (`GET {API_PREFIX}/regions/statuses` `box_states`
   *  serves the vocabulary): every box filter applies to the same box. */
  box_state?: string;
}

/** `browsePath` is the slot's declared browse collection
 *  (`capabilities.queue.browsePath`), so a slot never inherits another
 *  slot's route by accident. */
export async function getRegions(
  browsePath: string,
  params: RegionsQuery = {},
  signal?: AbortSignal,
): Promise<RegionsPage> {
  const page = await apiFetch<RegionsPage>(
    `${scoped()}${browsePath}${qs(params as Record<string, unknown>)}`,
    {},
    signal,
  );
  return {
    ...page,
    items: page.items.map((raw) => ({
      ...raw,
      slots: mapCropSlots(
        raw as unknown as Record<string, unknown>,
        (raw.bbox_norm ?? [0, 0, 0, 0]) as XYXY,
      ),
    })),
  };
}

/** Region-clustering background-job snapshot. The one-click pipeline result also
 *  carries the FP-rebuild + auto-assign sub-steps, and may report a re-partition
 *  that was skipped to protect a fresh manual refine (TTL). */
export interface RegionClusterJob {
  running: boolean;
  started_at: string | null;
  finished_at: string | null;
  result:
    | ({
        status?: string;
        n_regions?: number;
        n_clusters?: number;
        assigned?: number;
        fp_centroids?: { status: string; n_members?: number; k?: number };
        auto_fp?: { status: string; n_moved?: number; threshold?: number };
      } & Record<string, unknown>)
    | null;
  error: string | null;
}

/** Launch the one-click region-clustering pipeline (background job — a large
 *  pool takes minutes): rebuild FP sub-centroids → auto-pull tight FP matches →
 *  re-partition the good regions. Returns immediately; poll getRegionClusterStatus for completion. */
export function clusterRegions(
  maxRank?: number,
  opts: { forceRepartition?: boolean; autoFpThreshold?: number } = {},
  signal?: AbortSignal,
): Promise<RegionClusterJob> {
  return apiFetch(
    `${scoped()}${REGION_BASE}/cluster${qs({
      max_rank: maxRank,
      force_repartition: opts.forceRepartition,
      auto_fp_threshold: opts.autoFpThreshold,
    })}`,
    { method: 'POST' },
    signal,
  );
}

/** Poll the background region-clustering job. */
export function getRegionClusterStatus(signal?: AbortSignal): Promise<RegionClusterJob> {
  return apiFetch(`${scoped()}${REGION_BASE}/cluster/status`, {}, signal);
}

/** Per-bucket AHC refine over the region embeddings; writes region_cluster_subid. */
export function refineRegionCluster(
  clusterId: number,
  signal?: AbortSignal,
): Promise<{
  cluster_id: number;
  n_members: number;
  n_subclusters: number;
  action: string;
}> {
  return apiFetch(
    `${scoped()}${REGION_BASE}/clusters/refine/${clusterId}`,
    { method: 'POST' },
    signal,
  );
}

/** Region cluster cards (mirrors getClusters' Cluster shape). */
export function getRegionClusters(
  opts: { maxClusters?: number; perCluster?: number; maxRank?: number } = {},
  signal?: AbortSignal,
): Promise<{ clusters: Cluster[]; count: number }> {
  return apiFetch(
    `${scoped()}${REGION_BASE}/clusters${qs({
      max_clusters: opts.maxClusters,
      per_cluster: opts.perCluster,
      max_rank: opts.maxRank,
    })}`,
    {},
    signal,
  );
}

/** Background FP-centroid build-job snapshot + persisted centroid metadata. */
export interface RegionFpCentroidJob {
  running: boolean;
  started_at: string | null;
  finished_at: string | null;
  result: { status: string; n_members: number; k: number } | null;
  error: string | null;
  centroids: {
    trained_at: string | null;
    k: number | null;
    n_members: number | null;
  } | null;
}

/** (Re)build the FP centroid store — sub-types the FP bucket (background job). */
export function buildRegionFpCentroids(
  signal?: AbortSignal,
): Promise<RegionFpCentroidJob> {
  return apiFetch(
    `${scoped()}${REGION_BASE}/fp_centroids/build`,
    { method: 'POST' },
    signal,
  );
}

/** Poll the FP-centroid build job + read persisted centroid metadata. */
export function getRegionFpCentroidStatus(
  signal?: AbortSignal,
): Promise<RegionFpCentroidJob> {
  return apiFetch(`${scoped()}${REGION_BASE}/fp_centroids/status`, {}, signal);
}

export interface SuspectedFpItem extends RegionBrowseItem {
  suspected_fp_distance: number;
  nearest_fp_subid: string | null;
}

export interface SuspectedFpPage {
  items: SuspectedFpItem[];
  total: number;
  page: number;
  page_size: number;
  threshold?: number;
  centroids_built: boolean;
  trained_at?: string | null;
  message?: string;
}

/** Non-FP region crops ranked by similarity to the known FP centroids. */
export function getSuspectedFalsePositives(
  opts: { threshold?: number; page?: number; pageSize?: number } = {},
  signal?: AbortSignal,
): Promise<SuspectedFpPage> {
  return apiFetch(
    `${scoped()}${REGION_BASE}/suspected_false_positives${qs({
      threshold: opts.threshold,
      page: opts.page,
      page_size: opts.pageSize,
    })}`,
    {},
    signal,
  );
}

// A region training-candidate `?mode=` value. Not a closed union: the
// modes are served (`GET {API_PREFIX}/training_cohorts`), and the server
// rejects one it doesn't know.
export type TrainingCohortMode = string;

export function getTrainingCandidates(
  mode: TrainingCohortMode,
  params: { page?: number; page_size?: number; class_id?: number } = {},
  signal?: AbortSignal,
): Promise<RegionsPage> {
  return apiFetch<RegionsPage>(
    `${scoped()}${REGION_BASE}/training_candidates${qs({ mode, ...params })}`,
    {},
    signal,
  );
}

/**
 * `GET {API_PREFIX}/training_cohorts?class_id=` (2026-09-24 logic-moves
 * W6, item 13) — the deployment's own training-cohort definitions,
 * already resolved for the requested class (params fold `class_id` into
 * every cohort). `row_kind: 'region'` cohorts only appear when the
 * backend has a region profile configured; `class_id` omitted returns
 * the generic (unscoped) definitions. `/train`'s `runCohortQuery`
 * dispatches `endpoint`/`params` verbatim — no client re-derivation.
 */
export interface ServedTrainingCohort {
  id: string;
  label: string;
  description: string;
  cutoffs: Record<string, number>;
  /** Relative to API_PREFIX, e.g. `/crops`, `/regions/training_candidates`. */
  endpoint: string;
  /** Already resolved for the requested class — no `{classId}` templates. */
  params: Record<string, unknown>;
  row_kind: 'crop' | 'region';
}

export function getTrainingCohorts(
  classId?: number | null,
  signal?: AbortSignal,
): Promise<{ cohorts: ServedTrainingCohort[] }> {
  return apiFetch<{ cohorts: ServedTrainingCohort[] }>(
    `${scoped()}/training_cohorts${qs({ class_id: classId ?? undefined })}`,
    {},
    signal,
  );
}

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
 * A project's own served API prefix, verbatim. The only way to address a
 * project OTHER than the active one (the `/projects` page acts on each
 * row's own project): the prefix is the served
 * `ProjectSummary.prefix`, never assembled from a slug.
 */
export function projectPrefix(project: { prefix: string }): string {
  return project.prefix;
}

/** `GET {prefix}/pause`. A row action on `/projects`, so `global`: it
 *  belongs to that row's project, not the active one, and is never
 *  dropped as stale when the active project changes. */
export function getProjectPause(
  project: { prefix: string },
  signal?: AbortSignal,
): Promise<PipelinePauseState> {
  return apiFetch<PipelinePauseState>(`${projectPrefix(project)}/pause`, {}, signal, {
    global: true,
  });
}

/** `POST {prefix}/pause` — workers skip the project until resumed. */
export function pauseProject(project: { prefix: string }): Promise<PipelinePauseState> {
  return apiFetch<PipelinePauseState>(
    `${projectPrefix(project)}/pause`,
    { method: 'POST' },
    undefined,
    { global: true },
  );
}

/** `POST {prefix}/resume`. */
export function resumeProject(project: { prefix: string }): Promise<PipelinePauseState> {
  return apiFetch<PipelinePauseState>(
    `${projectPrefix(project)}/resume`,
    { method: 'POST' },
    undefined,
    { global: true },
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

/**
 * Pipeline-dashboard payload from `GET {API_PREFIX}/stats/dataset`. Contract
 * defined by `src/routers/curation/stats.py` — every nested key is
 * always present, numeric counters are always integers >= 0, and
 * `clusters.last_run_at` / `clusters.method` may be null when no
 * auto_label run has ever completed.
 */
export interface DatasetStats {
  as_of: string;
  total_crops: number;
  validated: number;
  test_holdout: number;
  by_source: Array<{ key: string; doc_count: number }>;
  labeled: {
    by_human: number;
    by_vlm: number;
    by_classifier: number;
    other: number;
  };
  regions: {
    /** Crops with a region_bbox_norm right now — the honest "crops with a
     *  region" count (matches the region cluster view). */
    boxed: number;
    /** Crops the verifier confirmed carry a real region (region_status='detected'). */
    confirmed: number;
    /** Sum of region_detector credit — includes rejected/failed attempts,
     *  so it OVERSTATES real regions. Not the headline. */
    total_detected: number;
    by_detector: number;
    by_segmenter: number;
    /** Crops where the operator drew a fresh region bbox from scratch. */
    by_human_drew: number;
    /** Crops whose region was verified by a human (the Confirm button). */
    verified_by_human: number;
    /** Crops whose region was verified by the VLM verifier (auto-verify). */
    verified_by_vlm: number;
    /**
     * Union: any region the operator touched — drew the bbox OR
     * confirmed an AI-proposed one. The dashboard surfaces this as
     * the honest "you reviewed N regions" number.
     */
    validated_by_human: number;
  };
  unlabeled: {
    pending_detection: number;
    pending_verification: number;
    /** Crops with no `class_id` at all, whatever their `label_source`
     *  (was miscounted as "labeled" — D1, visual audit 2026-09-24, before
     *  OpenProcessor #36 restricted `labeled.*` to docs with a real
     *  `class_id`). */
    no_label_source: number;
    /** Subset of `no_label_source` the VLM looked at but couldn't (or
     *  didn't) resolve to a class (#36 item 2). */
    vlm_no_class: number;
    /** F-23 (OpenProcessor d72cc63): crops a detector proposed but
     *  nothing has classified yet — a subset of `no_label_source`, like
     *  `vlm_no_class`. */
    by_proposal: number;
  };
  in_progress: {
    region_drain_total_unfinished: number;
    /** V-1 (OpenProcessor d72cc63): a served, human-readable line naming
     *  why the region drain can't progress (a region-profile dependency
     *  is down, and since when). Null when nothing is pending or every
     *  dependency is ready. Rendered verbatim. */
    region_stall_reason: string | null;
  };
  clusters: {
    last_run_at: string | null;
    /** Current total of distinct non-noise clusters in the index. */
    cluster_count: number;
    /** How many clusters the last auto-label run itself made (null: no run). */
    last_run_cluster_count?: number | null;
    residual_count: number;
    noise_count: number;
    method: string | null;
  };
}

export function getDatasetStats(signal?: AbortSignal): Promise<DatasetStats> {
  return apiFetch<DatasetStats>(`${scoped()}/stats/dataset`, {}, signal);
}

export async function getStats(signal?: AbortSignal): Promise<StatsSummary> {
  // The API returns
  //   {API_PREFIX}/stats/dataset:  {total_crops, validated, test_holdout, by_source}
  //   {API_PREFIX}/stats/classes:  {classes:[{class_id, class_name, count, validated_count}, ...]}
  // The labeler dashboard expects StatsSummary which uses validated_crops /
  // test_holdout_crops / per_class / ingestion.* — fold the two server
  // payloads into that shape so the dashboard can render directly.
  type RawDataset = {
    total_crops?: number;
    validated?: number;
    test_holdout?: number;
    by_source?: Array<{ key: string; doc_count: number }>;
  };
  type RawClasses = {
    classes: StatsSummary['per_class'];
    thresholds?: ClassThresholds;
  };
  // allSettled, not Promise.all: /stats/dataset can 503 (G1 — the live
  // op_items region_status mapping isn't aggregatable) while
  // /stats/classes is healthy. A dataset failure must still let
  // /export render its class table from per_class, so a rejection here
  // falls back to an empty RawDataset rather than sinking both calls.
  const [dsResult, clsResult] = await Promise.allSettled([
    apiFetch<RawDataset>(`${scoped()}/stats/dataset`, {}, signal),
    apiFetch<RawClasses>(`${scoped()}/stats/classes`, {}, signal),
  ]);
  const ds: RawDataset = dsResult.status === 'fulfilled' ? dsResult.value : {};
  const cls: RawClasses =
    clsResult.status === 'fulfilled' ? clsResult.value : { classes: [] };
  const totalImages = (ds.by_source ?? []).reduce(
    (acc, b) => acc + (b.doc_count || 0),
    0,
  );
  return {
    dataset_error:
      dsResult.status === 'rejected' ? (dsResult.reason as Error).message : null,
    total_crops: ds.total_crops ?? 0,
    validated_crops: ds.validated ?? 0,
    test_holdout_crops: ds.test_holdout ?? 0,
    ingestion: {
      images_processed: totalImages,
      images_pending: 0,
      last_run_at: null,
    },
    per_class: cls.classes.map((c) => ({
      class_id: c.class_id,
      class_name: c.class_name,
      count: c.count,
      validated_count: c.validated_count,
      adequacy: c.adequacy,
      aug_target: c.aug_target,
      aug_gap: c.aug_gap,
      trainable: c.trainable,
      trainable_gap: c.trainable_gap,
    })),
    thresholds: cls.thresholds,
  };
}

export async function getClasses(signal?: AbortSignal): Promise<ClassesResponse> {
  // Map the served `ClassEntry` rows to the labeler's RegistryClass shape
  // (`id`/`name`/`count`); `thresholds` and `reserved_hotkeys` pass through
  // verbatim — the server's own adequacy/hotkey rules, never recomputed
  // client-side.
  type RawClass = {
    class_id: number;
    class_name: string;
    group: string;
    sample_count: number;
    validated_count: number;
    cluster_size: number;
    deprecated: boolean;
    added_at: string | null;
    hotkey_letter: string | null;
    adequacy: string;
    kind: 'item' | 'region';
    trainable: number;
    trainable_gap: number;
    merged_into: number | null;
  };
  const res = await apiFetch<{
    classes: RawClass[];
    thresholds: ClassThresholds;
    reserved_hotkeys: string[];
  }>(`${scoped()}/classes`, {}, signal);
  const classes = res.classes.map((c) => ({
    id: c.class_id,
    name: c.class_name,
    group: c.group || null,
    count: c.sample_count,
    validated_count: c.validated_count,
    cluster_size: c.cluster_size,
    added_at: c.added_at ?? '',
    color: null,
    deprecated: c.deprecated,
    hotkey_letter: c.hotkey_letter,
    adequacy: c.adequacy,
    kind: c.kind,
    trainable: c.trainable,
    trainable_gap: c.trainable_gap,
    merged_into: c.merged_into,
  }));
  return { classes, thresholds: res.thresholds, reserved_hotkeys: res.reserved_hotkeys };
}

/** Raw cluster card from `{API_PREFIX}/clusters`. The backend is the single
 *  source of truth for every field — the frontend must never recompute
 *  dominant_class, purity, or is_unlabeled. */
type RawCluster = {
  cluster_id: number;
  cluster_kind: 'class' | 'candidate' | 'unassigned';
  size: number;
  validated_count: number;
  labelled_count: number;
  dominant_class_id: number | null;
  dominant_class_name: string | null;
  dominant_count: number;
  /** DQ-M2 fix (dq-queues cutover, 2026-09-24): the share of the
   *  `purity_n` members the cluster-geometry pass measured for this
   *  cluster whose NEAREST CLUSTER CENTROID is this cluster's own —
   *  independent of the labels that placed them, so a class cluster is
   *  no longer 1.0 by construction. Null when no member has been
   *  measured. */
  purity: number | null;
  /** How many members `purity` was computed over (the geometry pass's
   *  coverage for this cluster) — purity is noisy at low n. */
  purity_n: number | null;
  /** Always `'nearest_centroid'` today; served so the frontend never
   *  hardcodes what `purity` means. */
  purity_basis: string | null;
  /** Server-banded purity (see `purity_thresholds` below) — 'pure' | 'mixed' | 'noisy'. */
  purity_tier: 'pure' | 'mixed' | 'noisy' | null;
  /** Largest-class share among LABELLED members (the old label-based
   *  "purity" — always 1.0 for a class cluster by construction, which is
   *  exactly why it stopped being called `purity`). `promotable` uses
   *  this, not the geometry-based `purity` above. */
  label_purity: number | null;
  /** Share of this cluster's members that have any label at all. */
  labelled_share: number | null;
  /** Server's auto-promote eligibility gate for this cluster. */
  promotable: boolean;
  is_unlabeled: boolean;
  n_subclusters: number;
  updated_at: string | null;
  representatives: Array<{
    crop_id: string;
    cluster_distance: number | null;
    class_name: string | null;
    cluster_subid: string | null;
  }>;
};

type RawClustersResp = {
  items: RawCluster[];
  total: number;
  total_class_clusters: number;
  total_candidate_clusters: number;
  cluster_id_offset: number;
  /** Thresholds behind every item's `purity_tier`/`promotable` — informational,
   *  not re-applied client-side. */
  purity_thresholds?: {
    pure_min: number;
    mixed_min: number;
    promote_min_members: number;
    promote_min_labelled_share: number;
  } | null;
  /** Similarity floor behind every crop's `cluster_is_core` — copied onto
   *  each mapped `Cluster` so `/clusters/[id]`'s cut line never hardcodes it. */
  core_similarity_min?: number | null;
  /** D-4: the representatives window this response actually populated —
   *  echoed back so a caller windowing successive calls can advance past
   *  exactly what it got, not what it asked for. */
  representatives_offset?: number;
  representatives_limit?: number;
};

function _rawClusterToCluster(
  c: RawCluster,
  coreSimilarityMin: number | null = null,
): Cluster {
  return {
    id: c.cluster_id,
    cluster_kind: c.cluster_kind,
    size: c.size,
    validated_count: c.validated_count,
    dominant_class_id: c.dominant_class_id,
    dominant_class_name: c.dominant_class_name,
    // C1 (visual audit 2026-09-24): the subtitle's "dominant share" is
    // the served label-based share (largest class among labelled
    // members), never the nearest-centroid geometry `purity` below —
    // mapping `purity` here rendered "class_b · 3%" for a cluster that
    // is 616/616 class_b.
    dominant_pct: c.label_purity,
    dominant_count: c.dominant_count ?? null,
    labelled_count: c.labelled_count ?? null,
    purity: c.purity,
    purity_n: c.purity_n,
    purity_basis: c.purity_basis,
    purity_tier: c.purity_tier ?? null,
    label_purity: c.label_purity,
    labelled_share: c.labelled_share,
    promotable: !!c.promotable,
    core_similarity_min: coreSimilarityMin,
    is_unlabeled: c.is_unlabeled,
    representative_crop_ids: (c.representatives ?? []).map((r) => r.crop_id),
    has_subclusters: c.n_subclusters > 0,
    n_subclusters: c.n_subclusters,
    updated_at: c.updated_at,
  };
}

export interface ClustersResponse extends PaginatedResponse<Cluster> {
  /** D-4: the representatives window this response actually populated
   *  (echoed straight off the raw response) — a caller paging through
   *  windows advances by this, not by what it asked for. */
  representatives_offset: number;
  representatives_limit: number;
}

export async function getClusters(
  filter: ClusterFilter = {},
  signal?: AbortSignal,
): Promise<ClustersResponse> {
  // Single round-trip for the full (size-desc, size-capped-by-max_clusters)
  // card list — the backend's {API_PREFIX}/clusters aggregation already
  // returns dominant class, purity, purity_tier, promotable,
  // validated_count, n_subclusters, cluster_kind, and is_unlabeled for
  // EVERY card regardless of the representatives window. The frontend
  // ONLY shapes the result into the labeler's Cluster type — no semantic
  // compute here.
  //
  // D-4 (docs/design/curation_query_performance_audit.md): representatives
  // (the per-card thumbnail crops, one `_msearch` each) are windowed by
  // `offset`/`limit` — cards outside `[offset, offset+limit)` of the
  // returned card list come back with `representatives: []`. The caller
  // is responsible for requesting only the window it's actually going to
  // render (see /clusters' `loadMoreRepresentatives`), not every card, so
  // scrolling past the first screenful doesn't re-run the aggregation's
  // representative lookup for clusters nobody has scrolled to yet.
  const raw = await apiFetch<RawClustersResp>(
    `${scoped()}/clusters${qs({
      per_cluster: filter.per_cluster ?? 4,
      class_id: filter.class_id ?? undefined,
      // DQ-M4: lets the caller fetch one card's representatives directly
      // by id, independent of the size-desc `offset`/`limit` window —
      // see ClusterFilter.cluster_id's doc comment.
      cluster_id: filter.cluster_id ?? undefined,
      // Pull enough buckets that the 512 IVF candidate clusters (+ class
      // clusters) all come back in one call — the endpoint returns them
      // ordered by size, not paginated, so a low cap would silently drop
      // the smaller candidate buckets from the "Unlabeled only" view.
      max_clusters: 2000,
      // Primary-subject grid filters — card stats reflect only passing crops.
      max_rank: filter.max_rank ?? undefined,
      min_blur_ratio: filter.min_blur_ratio ?? undefined,
      class_source: filter.class_source ?? undefined,
      offset: filter.representatives_offset ?? undefined,
      limit: filter.representatives_limit ?? undefined,
    })}`,
    {},
    signal,
  );
  const coreSimilarityMin = raw.core_similarity_min ?? null;
  const items = (raw.items ?? []).map((c) => _rawClusterToCluster(c, coreSimilarityMin));
  return {
    items,
    total: raw.total ?? items.length,
    page: 1,
    page_size: items.length,
    purity_thresholds: raw.purity_thresholds ?? null,
    representatives_offset:
      raw.representatives_offset ?? filter.representatives_offset ?? 0,
    representatives_limit:
      raw.representatives_limit ?? filter.representatives_limit ?? 50,
  };
}

/** Convert API's [x1, y1, x2, y2] to the labeler's BBoxNorm {cx, cy, w, h}. */
function xyxyToBBoxNorm(bb: number[]): import('./types').BBoxNorm {
  const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = bb;
  return {
    cx: (x1 + x2) / 2,
    cy: (y1 + y2) / 2,
    w: Math.max(0, x2 - x1),
    h: Math.max(0, y2 - y1),
  };
}

/** Raw crop shape from the {API_PREFIX}/crops API. */
export type RawCrop = {
  crop_id: string;
  image_id?: string;
  image_path: string;
  bbox_norm: number[];
  class_id?: number | null;
  class_name?: string | null;
  class_source?: string;
  confidence?: number;
  /** Confidence of whoever set the label: VLM high/medium/low map to
   *  0.92/0.70/0.40, classifier labels carry their own score, human
   *  labels are null (dq-queues cutover, 2026-09-24). Distinct from
   *  `confidence`, which is always the detector/classifier score. */
  class_confidence?: number | null;
  /** `'vlm' | 'model' | null` — which pipeline set `class_confidence`. */
  class_confidence_source?: string | null;
  /** The VLM's verbatim class answer, even when it didn't match a
   *  registry class or wasn't auto-applied. */
  vlm_raw_class?: string | null;
  vlm_class_attempted_at?: string | null;
  /** `no_answer | no_match | invalid_index | unparseable | null`. */
  vlm_class_empty_reason?: string | null;
  cluster_id?: number | null;
  cluster_distance?: number | null;
  /** Nearest cluster centroid id when it differs from `cluster_id` (the
   *  item has left its cluster) — null together with cluster_distance/
   *  cluster_similarity/cluster_is_core in that case. */
  cluster_nearest_id?: number | null;
  /** Server-computed cosine similarity to this crop's cluster centroid
   *  (0..1) — the served replacement for the old client `1 -
   *  cluster_distance` estimate. Null when the backend hasn't computed
   *  it for this crop. */
  cluster_similarity?: number | null;
  /** Server-computed: `cluster_similarity >= core_similarity_min`. Drives
   *  the cluster-detail cut line — see `Crop.cluster_is_core`. */
  cluster_is_core?: boolean | null;
  cluster_subid?: string | null;
  label_validated?: boolean;
  /** G2: the class-label-specific validation flag. `label_validated` is
   *  `class_validated OR region_validated` server-side and over-reports
   *  a validated class. */
  class_validated?: boolean;
  label_source?: string;
  class_detector?: string | null;
  class_detector_version?: string | null;
  class_labeled_at?: string | null;
  class_labeler?: string | null;
  /** Ingest source tag — replaces the dead `hdd_source` (2026-09-24
   *  logic-moves cutover, item 14/G3). */
  source?: string | null;
  test_holdout?: boolean;
  crop_rank_in_image?: number | null;
  crop_area_norm?: number | null;
  blur_lap_ratio?: number | null;
  classifier_raw_confidence?: number | null;
  proposal_name?: string | null;
  vlm_confidence?: string | null;
  vlm_proposed_class_id?: number | null;
  vlm_proposed_class_name?: string | null;
  /** The backend's confirmable suggestion — served on every crop-shaped
   *  item (item 11, 2026-09-24), not just review-queue rows. */
  proposed_class_id?: number | null;
  proposed_class_name?: string | null;
  // Curation scores (Phase 3, docs/curation-strategy-plan-2026-09.md §4).
  // Optional/forward-tolerant: an un-backfilled pool just omits these.
  mistakenness_score?: number | null;
  mistakenness_method?: string | null;
  mistakenness_version?: string | null;
  mistakenness_scored_at?: string | null;
  // F8 D1 (OpenProcessor 51b05d7): the probe's opinion on this item.
  probe_disagreement?: boolean | null;
  probe_in_scope?: boolean | null;
  probe_model_version?: string | null;
  // OpenProcessor main 8990ede: server-computed actionability, folding
  // in scope + disagreement + the server's own confidence threshold.
  probe_actionable?: boolean | null;
  thumbnail_url?: string;
  updated_at?: string;
  class_excluded?: boolean;
  excluded_reason?: string | null;
  excluded_at?: string | null;
  item_text_lines?: unknown;
  // W9 / W10 / P4 item provenance (contract f582aa05 `ItemDoc`); each is
  // served on every item, a missing key maps to null / false / [].
  vlm_endpoint?: string | null;
  vlm_model?: string | null;
  vlm_prompt_pack?: string | null;
  label_locked?: boolean;
  import_ids?: string[];
  dataset_split?: string | null;
  imported_at?: string | null;
  proposed_by_import?: string | null;
  on_negative_frame?: boolean;
  import_standalone_region?: boolean;
  proposal_chain?: string[];
  origin_project?: string | null;
  origin_item_id?: string | null;
  origin_image_id?: string | null;
  origin_split?: string | null;
  combine_conflict?: boolean;
  combine_conflict_origins?: string[];
  combine_merged_origins?: string[];
};

/**
 * Every key `RawCrop` declares, as a runtime value. `src/lib/test/makeItem.ts`
 * builds a fixture from it that can't silently miss a field, and
 * `src/lib/contract/wireKeys.test.ts` checks it against the backend's
 * vendored item-wire key list. The `satisfies` + exhaustiveness check below
 * makes an added or removed `RawCrop` key a compile error here.
 */
export const RAW_CROP_KEYS = [
  'crop_id',
  'image_id',
  'image_path',
  'bbox_norm',
  'class_id',
  'class_name',
  'class_source',
  'confidence',
  'class_confidence',
  'class_confidence_source',
  'vlm_raw_class',
  'vlm_class_attempted_at',
  'vlm_class_empty_reason',
  'cluster_id',
  'cluster_distance',
  'cluster_nearest_id',
  'cluster_similarity',
  'cluster_is_core',
  'cluster_subid',
  'label_validated',
  'class_validated',
  'label_source',
  'class_detector',
  'class_detector_version',
  'class_labeled_at',
  'class_labeler',
  'source',
  'test_holdout',
  'crop_rank_in_image',
  'crop_area_norm',
  'blur_lap_ratio',
  'classifier_raw_confidence',
  'proposal_name',
  'vlm_confidence',
  'vlm_proposed_class_id',
  'vlm_proposed_class_name',
  'proposed_class_id',
  'proposed_class_name',
  'mistakenness_score',
  'mistakenness_method',
  'mistakenness_version',
  'mistakenness_scored_at',
  'probe_disagreement',
  'probe_in_scope',
  'probe_model_version',
  'probe_actionable',
  'thumbnail_url',
  'updated_at',
  'class_excluded',
  'excluded_reason',
  'excluded_at',
  'item_text_lines',
  'vlm_endpoint',
  'vlm_model',
  'vlm_prompt_pack',
  'label_locked',
  'import_ids',
  'dataset_split',
  'imported_at',
  'proposed_by_import',
  'on_negative_frame',
  'import_standalone_region',
  'proposal_chain',
  'origin_project',
  'origin_item_id',
  'origin_image_id',
  'origin_split',
  'combine_conflict',
  'combine_conflict_origins',
  'combine_merged_origins',
] as const satisfies readonly (keyof RawCrop)[];
// Compile error if RAW_CROP_KEYS drops (or never gains) a RawCrop key.
type _RawCropKeysExhaustive =
  Exclude<keyof RawCrop, (typeof RAW_CROP_KEYS)[number]> extends never
    ? true
    : [
        'RAW_CROP_KEYS is missing',
        Exclude<keyof RawCrop, (typeof RAW_CROP_KEYS)[number]>,
      ];
const _rawCropKeysExhaustive: _RawCropKeysExhaustive = true;

/** Parses the wire `ItemTextLine[]` tolerantly — a malformed/absent entry
 *  is dropped rather than throwing, since this is OCR output the backend
 *  may not have backfilled for every item. */
function asItemTextLines(v: unknown): ItemTextLine[] {
  if (!Array.isArray(v)) return [];
  const out: ItemTextLine[] = [];
  for (const raw of v) {
    if (typeof raw !== 'object' || raw === null) continue;
    const r = raw as Record<string, unknown>;
    out.push({
      text: typeof r.text === 'string' ? r.text : null,
      confidence: typeof r.confidence === 'number' ? r.confidence : null,
      box_norm: Array.isArray(r.box_norm) ? (r.box_norm as number[]) : null,
      rel_height: typeof r.rel_height === 'number' ? r.rel_height : null,
    });
  }
  return out;
}

/** A served string list, tolerantly: anything else (absent, null) is `[]`. */
function asStringArray(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
}

export function mapRawCrop(c: RawCrop): Crop {
  const bb = c.bbox_norm ?? [0, 0, 0, 0];
  const out: Crop = {
    id: c.crop_id,
    source_image_path: c.image_path,
    image_id: c.image_id || undefined,
    bbox_norm: xyxyToBBoxNorm(bb),
    class_id: c.class_id ?? null,
    class_name: c.class_name ?? null,
    class_source: c.class_source ?? null,
    label_source: c.label_source || 'unknown',
    label_validated: !!c.label_validated,
    class_validated: !!c.class_validated,
    label_confidence: c.confidence ?? null,
    class_confidence: c.class_confidence ?? null,
    class_confidence_source: c.class_confidence_source ?? null,
    vlm_raw_class: c.vlm_raw_class ?? null,
    vlm_class_attempted_at: c.vlm_class_attempted_at ?? null,
    vlm_class_empty_reason: c.vlm_class_empty_reason ?? null,
    cluster_id: c.cluster_id ?? null,
    cluster_distance: c.cluster_distance ?? null,
    cluster_nearest_id: c.cluster_nearest_id ?? null,
    // Served directly — no client 1-cosine-distance estimate.
    similarity_to_centroid: c.cluster_similarity ?? null,
    cluster_is_core: c.cluster_is_core ?? null,
    cluster_subid: c.cluster_subid ?? null,
    class_detector: c.class_detector ?? null,
    class_detector_version: c.class_detector_version ?? null,
    class_labeled_at: c.class_labeled_at ?? null,
    class_labeler: c.class_labeler ?? null,
    source: c.source ?? null,
    proposed_class_id: c.proposed_class_id ?? null,
    proposed_class_name: c.proposed_class_name ?? null,
    test_holdout: !!c.test_holdout,
    crop_rank_in_image: c.crop_rank_in_image ?? null,
    crop_area_norm: c.crop_area_norm ?? null,
    blur_lap_ratio: c.blur_lap_ratio ?? null,
    classifier_raw_confidence: c.classifier_raw_confidence ?? null,
    proposal_name: c.proposal_name ?? null,
    vlm_confidence: c.vlm_confidence ?? null,
    vlm_suggested_class_id: c.vlm_proposed_class_id ?? null,
    vlm_suggested_class_name: c.vlm_proposed_class_name ?? null,
    mistakenness_score: c.mistakenness_score ?? null,
    mistakenness_method: c.mistakenness_method ?? null,
    mistakenness_version: c.mistakenness_version ?? null,
    mistakenness_scored_at: c.mistakenness_scored_at ?? null,
    probe_disagreement: c.probe_disagreement ?? null,
    probe_in_scope: c.probe_in_scope ?? null,
    probe_model_version: c.probe_model_version ?? null,
    probe_actionable: c.probe_actionable ?? null,
    class_excluded: !!c.class_excluded,
    excluded_reason: c.excluded_reason ?? null,
    excluded_at: c.excluded_at ?? null,
    item_text_lines: asItemTextLines(c.item_text_lines),
    vlm_endpoint: c.vlm_endpoint ?? null,
    vlm_model: c.vlm_model ?? null,
    vlm_prompt_pack: c.vlm_prompt_pack ?? null,
    label_locked: !!c.label_locked,
    import_ids: asStringArray(c.import_ids),
    dataset_split: c.dataset_split ?? null,
    imported_at: c.imported_at ?? null,
    proposed_by_import: c.proposed_by_import ?? null,
    on_negative_frame: !!c.on_negative_frame,
    import_standalone_region: !!c.import_standalone_region,
    proposal_chain: asStringArray(c.proposal_chain),
    origin_project: c.origin_project ?? null,
    origin_item_id: c.origin_item_id ?? null,
    origin_image_id: c.origin_image_id ?? null,
    origin_split: c.origin_split ?? null,
    combine_conflict: !!c.combine_conflict,
    combine_conflict_origins: asStringArray(c.combine_conflict_origins),
    combine_merged_origins: asStringArray(c.combine_merged_origins),
    slots: mapCropSlots(c as unknown as Record<string, unknown>, bb as XYXY),
    // Preserve server-side updated_at — overriding it client-side breaks
    // ordering and lets the same crop key appear twice in keyed each blocks
    // (Svelte each_key_duplicate).
    updated_at:
      typeof (c as Record<string, unknown>).updated_at === 'string'
        ? ((c as Record<string, unknown>).updated_at as string)
        : '',
  };
  return out;
}

export async function getCluster(
  id: number,
  page = 1,
  pageSize = 60,
  signal?: AbortSignal,
  opts: {
    classSource?: string | null;
    maxRank?: number | null;
    minBlurRatio?: number | null;
    classifierConfLt?: number | null;
    /**
     * Forwarded verbatim to `{API_PREFIX}/crops?order=`. Only `'outliers'` is
     * special-cased server-side today (crops.py `order` query param —
     * see docs/curation-strategy-plan-2026-09.md §1); an id the backend
     * doesn't recognize is harmless (qs() still sends it, the server
     * just falls back to its default ordering). Typed as `string` rather
     * than a fixed union so a new `{API_PREFIX}/methods`-reported order id doesn't
     * require touching this signature — callers should still gate which
     * ids they actually offer against what `{API_PREFIX}/methods` reports.
     */
    order?: string | null;
    /**
     * Pool-scale overlay parameter, forwarded to `{API_PREFIX}/crops?k=` only when
     * set (curation-strategy plan Phase 4 — `order: 'diverse'`'s "how
     * many diverse crops" count). Meaningless for every other `order`
     * value; the caller (`/clusters/[id]`) only sets it in diverse mode.
     */
    k?: number | null;
  } = {},
): Promise<{ cluster: Cluster; crops: PaginatedResponse<Crop> }> {
  // Two parallel calls: paginated crops + the authoritative cluster
  // card from {API_PREFIX}/clusters (server-computed). The page no longer
  // derives any of the cluster's identity fields client-side.
  //
  // `classSource` narrows the crop grid to one source bucket
  // (v6_model / gemma / human / v6_low_conf / ...) without touching
  // the cluster card stats — the header still shows the whole-cluster
  // totals so the operator sees the filter against the full size.
  type CropPage = {
    total: number;
    page: number;
    page_size: number;
    crops: RawCrop[];
    /** Only present for a pool-scale overlay ordering (e.g.
     *  `order=diverse`) — see `PaginatedResponse.order_method` /
     *  `.order_version` / `.n_pool` in types.ts. Untyped/optional and
     *  parsed tolerantly below: the shape belongs to whichever `order`
     *  overlay is active, not something this function should assume. */
    method?: unknown;
    version?: unknown;
    n_pool?: unknown;
  };
  const cropQuery: Record<string, unknown> = {
    cluster_id: id,
    page,
    page_size: pageSize,
  };
  if (opts.classSource) cropQuery.class_source = opts.classSource;
  if (opts.maxRank != null) cropQuery.max_rank = opts.maxRank;
  if (opts.minBlurRatio != null) cropQuery.min_blur_ratio = opts.minBlurRatio;
  if (opts.classifierConfLt != null) cropQuery.classifier_conf_lt = opts.classifierConfLt;
  if (opts.order) cropQuery.order = opts.order;
  if (opts.k != null) cropQuery.k = opts.k;
  const [cropPage, clustersResp] = await Promise.all([
    apiFetch<CropPage>(`${scoped()}/crops${qs(cropQuery)}`, {}, signal),
    apiFetch<RawClustersResp>(
      // cluster_id (not class_id!) is the correct filter for "fetch this
      // one cluster's card by its own identity" — cluster_id == class_id
      // is only an invariant for 'class' clusters, so filtering by
      // class_id silently returned nothing for every 'candidate' cluster
      // (id >= RESIDUAL_CLUSTER_ID_OFFSET, no matching class exists),
      // which fell back to the null-identity stub below and showed no
      // human-readable name in the header even though {API_PREFIX}/clusters'
      // list view has dominant_class_name for the same cluster.
      `${scoped()}/clusters${qs({ per_cluster: 4, max_clusters: 1, cluster_id: id })}`,
      {},
      signal,
    ).catch(() => null),
  ]);
  const items = cropPage.crops.map(mapRawCrop);
  const found = clustersResp?.items?.find((c) => c.cluster_id === id) ?? null;
  const cluster: Cluster = found
    ? _rawClusterToCluster(found, clustersResp?.core_similarity_min ?? null)
    : {
        // Fallback only if the cluster card lookup failed — leaves
        // identity fields null but lets the crop grid render.
        id,
        cluster_kind: 'unassigned',
        size: cropPage.total,
        validated_count: 0,
        dominant_class_id: null,
        dominant_class_name: null,
        dominant_pct: null,
        purity: null,
        purity_tier: null,
        promotable: false,
        core_similarity_min: null,
        is_unlabeled: true,
        has_subclusters: false,
        n_subclusters: 0,
        representative_crop_ids: items.slice(0, 4).map((c) => c.id),
        updated_at: null,
      };
  return {
    cluster,
    crops: {
      items,
      total: cropPage.total,
      page: cropPage.page,
      page_size: cropPage.page_size,
      order_method: typeof cropPage.method === 'string' ? cropPage.method : null,
      order_version: typeof cropPage.version === 'string' ? cropPage.version : null,
      n_pool: typeof cropPage.n_pool === 'number' ? cropPage.n_pool : null,
    },
  };
}

export async function getCrops(
  filter: CropFilter = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<Crop>> {
  type Raw = { total: number; page: number; page_size: number; crops: RawCrop[] };
  const raw = await apiFetch<Raw>(`${scoped()}/crops${qs({ ...filter })}`, {}, signal);
  return {
    items: raw.crops.map(mapRawCrop),
    total: raw.total,
    page: raw.page,
    page_size: raw.page_size,
  };
}

/**
 * `GET {API_PREFIX}/crops/{id}/history` — the item's label-write history,
 * oldest first. Untyped on the wire beyond `crop_id`/`entries`
 * (`additionalProperties: true`); each entry's other keys are read
 * tolerantly by the caller (`CropHistoryEntry`'s fields are all
 * optional). Used by `CropMetaPanel`'s lazy-on-open history section (G7).
 */
export function getCropHistory(
  cropId: string,
  signal?: AbortSignal,
): Promise<CropHistoryResponse> {
  return apiFetch<CropHistoryResponse>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/history`,
    {},
    signal,
  );
}

/**
 * `GET {API_PREFIX}/crops/{id}/context` — the crop's shared source image
 * metadata plus every item cropped from it (siblings, including the
 * requested crop). The image itself is served by `/crops/{id}/image`
 * (`getSourceImageScaled`/`getSourceImageFull`) — as of K6 this is a
 * clean image with no server-drawn boxes; `SourceImageOverlay.svelte`
 * draws every box/label from this response's `items`.
 */
export async function getCropContext(
  cropId: string,
  signal?: AbortSignal,
): Promise<CropContextResponse> {
  type Raw = {
    image: CropContextResponse['image'];
    items: RawCrop[];
  };
  const raw = await apiFetch<Raw>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/context`,
    {},
    signal,
  );
  return { image: raw.image, items: raw.items.map(mapRawCrop) };
}

export function putCropLabel(
  cropId: string,
  classId: number,
  signal?: AbortSignal,
): Promise<Crop> {
  return apiFetch<Crop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/label`,
    {
      method: 'PUT',
      body: JSON.stringify({ class_id: classId, validated: true }),
    },
    signal,
  );
}

export function bulkLabel(
  cropIds: string[],
  classId: number,
  signal?: AbortSignal,
): Promise<BulkLabelResult> {
  // Backend route is PUT (matches the single-crop /label PUT shape).
  return apiFetch<BulkLabelResult>(
    `${scoped()}/crops/batch_label`,
    {
      method: 'PUT',
      body: JSON.stringify({ crop_ids: cropIds, class_id: classId, validated: true }),
    },
    signal,
  );
}

/**
 * Undo the crop's most recent human class write. The backend restores
 * its own snapshot (earlier human label, VLM suggestion, ingest proposal
 * or unlabeled) and returns the restored item. `409` means nothing is
 * left to undo.
 */
export async function undoCropLabel(cropId: string, signal?: AbortSignal): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/label/undo`,
    { method: 'POST' },
    signal,
  );
  return mapRawCrop(raw);
}

/**
 * OpenProcessor d72cc63: the batch write bodies (`/ingest/batch`,
 * `/crops/{label,region}/undo_batch`, `/crops/discard_batch`, the VLM
 * batch routes) are `extra='forbid'` with a required non-empty list, so
 * an empty list is a guaranteed 422. Refuse it here, before any request
 * goes out — callers disable the action instead of sending it.
 */
export function assertNonEmptyBatch(what: string, ids: readonly unknown[]): void {
  if (ids.length === 0) {
    throw new Error(`${what}: nothing selected — no request sent`);
  }
}

/**
 * `POST {API_PREFIX}/crops/label/undo_batch` — batch form of
 * `undoCropLabel`: each crop is restored independently to its own state
 * before its most recent human class write, so undoing a `bulkLabel` or
 * `resolveNewClassProposal` means passing the same ids that write
 * reported in `updated_ids`. `409` when no crop in the batch had
 * anything to undo.
 */
export async function undoLabelBatch(
  cropIds: string[],
  signal?: AbortSignal,
): Promise<CropUndoBatchResult> {
  assertNonEmptyBatch('undo', cropIds);
  type Raw = {
    items?: RawCrop[];
    undone?: number;
    nothing_to_undo?: string[];
    conflicts?: string[];
    not_found?: string[];
  };
  const raw = await apiFetch<Raw>(
    `${scoped()}/crops/label/undo_batch`,
    { method: 'POST', body: JSON.stringify({ crop_ids: cropIds }) },
    signal,
  );
  return {
    items: (raw.items ?? []).map(mapRawCrop),
    undone: raw.undone ?? 0,
    nothing_to_undo: raw.nothing_to_undo ?? [],
    conflicts: raw.conflicts ?? [],
    not_found: raw.not_found ?? [],
  };
}

/** Options shared by the single and batch discard endpoints. */
export interface DiscardOptions {
  /** Clear the crop's class (it doesn't belong where it is). Default true. */
  clear_class?: boolean;
  /** Also stamp review_dismissed_at, so it never resurfaces in /review. Default false. */
  dismiss_from_review?: boolean;
}

/**
 * Discard a single crop. Recorded like a label write, so
 * `POST /crops/{id}/label/undo` reverses it — callers should push an
 * undo entry for the crop on success.
 */
export async function discardCrop(
  cropId: string,
  opts: DiscardOptions = {},
  signal?: AbortSignal,
): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/discard`,
    { method: 'POST', body: JSON.stringify(opts) },
    signal,
  );
  return mapRawCrop(raw);
}

export interface DiscardBatchResult {
  items: Crop[];
  discarded: number;
  conflicts: BulkLabelConflict[];
  not_found: string[];
}

/** Bulk variant of `discardCrop`. `items` carries only the crops actually
 *  discarded — conflicted/not-found ids are reported separately. */
export async function discardCropsBatch(
  cropIds: string[],
  opts: DiscardOptions = {},
  signal?: AbortSignal,
): Promise<DiscardBatchResult> {
  assertNonEmptyBatch('discard', cropIds);
  const raw = await apiFetch<{
    items: RawCrop[];
    discarded: number;
    conflicts: BulkLabelConflict[];
    not_found: string[];
  }>(
    `${scoped()}/crops/discard_batch`,
    { method: 'POST', body: JSON.stringify({ crop_ids: cropIds, ...opts }) },
    signal,
  );
  return { ...raw, items: raw.items.map(mapRawCrop) };
}

/**
 * Permanently dismiss a crop from every /review queue.
 *
 * Stamps ``review_dismissed_at`` + ``review_dismissed_by='human'`` on
 * the crop. The server's review_queue handler must_not's any crop with
 * ``review_dismissed_at``, so dismissed crops never reappear until
 * `reviewUndismissCrop` clears it. The crop's class / region state is
 * left intact — only review visibility changes.
 */
export function reviewDismissCrop(cropId: string, signal?: AbortSignal): Promise<void> {
  return apiFetch<void>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/review_dismiss`,
    { method: 'POST' },
    signal,
  );
}

/** Reverses `reviewDismissCrop`, returning the restored item. */
export async function reviewUndismissCrop(
  cropId: string,
  signal?: AbortSignal,
): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/review_undismiss`,
    { method: 'POST' },
    signal,
  );
  return mapRawCrop(raw);
}

/**
 * Dismiss the VLM's proposed class on a crop ("reject VLM suggestion").
 * The backend clears `vlm_proposed_class_*` and returns the updated
 * item. `409` means the crop had no suggestion to dismiss.
 */
export async function vlmDismissCrop(
  cropId: string,
  signal?: AbortSignal,
): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/vlm_dismiss`,
    { method: 'POST' },
    signal,
  );
  return mapRawCrop(raw);
}

/**
 * `POST {API_PREFIX}/crops/{id}/vlm_dismiss/undo` (M6, backend b654da5):
 * put `vlm_dismissed_*` back to its state before the latest
 * `vlm_dismiss`, so the dismissed suggestion is live again. `409` means
 * there was no dismissal to undo.
 */
export async function undoVlmDismiss(
  cropId: string,
  signal?: AbortSignal,
): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/vlm_dismiss/undo`,
    { method: 'POST' },
    signal,
  );
  return mapRawCrop(raw);
}

/**
 * `POST {API_PREFIX}/crops/{id}/region/undo` (M6, backend b654da5):
 * restore the region to its state before the most recent not-yet-undone
 * human region write (confirm, reject, false positive, box edit, status
 * or text change) — repeated calls step back further. Class fields are
 * untouched. `409` means nothing is left to undo.
 */
export async function undoCropRegion(
  cropId: string,
  signal?: AbortSignal,
): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/region/undo`,
    { method: 'POST' },
    signal,
  );
  return mapRawCrop(raw);
}

/**
 * `POST {API_PREFIX}/crops/region/undo_batch` — batch form of
 * `undoCropRegion`: each crop is restored independently, so undoing a
 * `PUT /crops/batch_region` or `POST /regions/batch_status` means passing
 * the same ids that write touched. `409` when no crop in the batch had
 * anything to undo.
 */
export async function undoCropRegionBatch(
  cropIds: string[],
  signal?: AbortSignal,
): Promise<CropRegionUndoBatchResult> {
  assertNonEmptyBatch('undo', cropIds);
  type Raw = {
    items?: RawCrop[];
    undone?: number;
    nothing_to_undo?: string[];
    conflicts?: string[];
    not_found?: string[];
  };
  const raw = await apiFetch<Raw>(
    `${scoped()}/crops/region/undo_batch`,
    { method: 'POST', body: JSON.stringify({ crop_ids: cropIds }) },
    signal,
  );
  return {
    items: (raw.items ?? []).map(mapRawCrop),
    undone: raw.undone ?? 0,
    nothing_to_undo: raw.nothing_to_undo ?? [],
    conflicts: raw.conflicts ?? [],
    not_found: raw.not_found ?? [],
  };
}

/* ------------------------------------------------------------------ */
/* W8 multi-box region writes (lockstep with the backend's W8; see     */
/* docs/design/w8-multibox-frontend-plan-2026-09-26.md). Element shapes */
/* are RegionBoxInput from annotations/multiBox.ts.                     */
/* ------------------------------------------------------------------ */

/** `PUT /crops/{crop_id}/regions` (W8.8) — replaces the box list on one
 *  crop. `regionStatus` optionally applies a whole-set status to the
 *  built list in the same write (Enter-after-edit: `'detected'`). */
export async function putRegionBoxes(
  cropId: string,
  boxes: RegionBoxInput[],
  opts: { regionStatus?: string; expectedRegionRevision?: number } = {},
  signal?: AbortSignal,
): Promise<Crop> {
  const body: Record<string, unknown> = {
    boxes,
    frame: 'parent',
    region_label_source: 'human',
  };
  if (opts.regionStatus != null) body.region_status = opts.regionStatus;
  if (opts.expectedRegionRevision != null) {
    body.expected_region_revision = opts.expectedRegionRevision;
  }
  const raw = await apiFetch<{ item: RawCrop }>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/regions`,
    { method: 'PUT', body: JSON.stringify(body) },
    signal,
  );
  return mapRawCrop(raw.item);
}

/** `PUT /crops/batch_regions` (W8.8) — replaces every listed crop's box
 *  list with the SAME new boxes (every element must be `box_id: null`;
 *  typically `boxes: []`, "none visible"). */
export async function putBatchRegions(
  cropIds: string[],
  boxes: Array<{ bbox_norm: [number, number, number, number]; state?: string }>,
  opts: { regionStatus?: string } = {},
  signal?: AbortSignal,
): Promise<void> {
  assertNonEmptyBatch('batch region replace', cropIds);
  const body: Record<string, unknown> = { crop_ids: cropIds, boxes };
  if (opts.regionStatus != null) body.region_status = opts.regionStatus;
  await apiFetch(
    `${scoped()}/crops/batch_regions`,
    { method: 'PUT', body: JSON.stringify(body) },
    signal,
  );
}

/** `PATCH /crops/{crop_id}/regions/{box_id}` (W8.8) — per-box state/text
 *  flip. Used by the selected-box accept/reject keymap actions. */
export async function patchRegionBox(
  cropId: string,
  boxId: string,
  patch: { state?: string; text?: string | null; expectedRegionRevision?: number },
  signal?: AbortSignal,
): Promise<Crop> {
  const body: Record<string, unknown> = {};
  if (patch.state != null) body.state = patch.state;
  if (patch.text !== undefined) body.text = patch.text;
  if (patch.expectedRegionRevision != null) {
    body.expected_region_revision = patch.expectedRegionRevision;
  }
  const raw = await apiFetch<{ item: RawCrop }>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/regions/${encodeURIComponent(boxId)}`,
    { method: 'PATCH', body: JSON.stringify(body) },
    signal,
  );
  return mapRawCrop(raw.item);
}

/** A stale `expected_region_revision` is `409 region_conflict` on the
 *  per-item box writes: the body carries the current revision, the
 *  current box ids and the current `item`, which the caller adopts
 *  instead of re-deriving anything. */
export interface RegionConflictDetail {
  currentRegionRevision: number;
  currentBoxIds: string[];
  item: Crop;
}

export function regionConflictDetail(e: unknown): RegionConflictDetail | null {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (d.error !== 'region_conflict' || !d.item || typeof d.item !== 'object') return null;
  return {
    currentRegionRevision:
      typeof d.current_region_revision === 'number' ? d.current_region_revision : 0,
    currentBoxIds: Array.isArray(d.current_box_ids)
      ? d.current_box_ids.filter((x): x is string => typeof x === 'string')
      : [],
    item: mapRawCrop(d.item as RawCrop),
  };
}

export interface RegionBatchConflict {
  crop_id: string;
  error: string;
  message: string;
  current_source: string | null;
  current_region_revision: number;
  current_box_ids: string[];
  item: RawCrop;
}

export interface RegionBatchBoxStateResult {
  updated: number;
  invalid: Array<{ crop_id: string; box_id: string; error: string; message: string }>;
  conflicts: RegionBatchConflict[];
  items: Crop[];
}

/** `POST /regions/batch_box_state` (W8.8) — one state on many boxes
 *  across items (region-gallery triage / a region cluster = a set of
 *  boxes). Never flips a whole item's other boxes — use `batchRegionStatus`
 *  for that. */
export async function postBatchBoxState(
  targets: Array<{ cropId: string; boxId: string }>,
  state: string,
  signal?: AbortSignal,
): Promise<RegionBatchBoxStateResult> {
  if (targets.length === 0)
    throw new Error('postBatchBoxState requires at least one target');
  type Raw = {
    updated: number;
    invalid?: Array<{ crop_id: string; box_id: string; error: string; message: string }>;
    conflicts?: RegionBatchConflict[];
    items?: RawCrop[];
  };
  const raw = await apiFetch<Raw>(
    `${scoped()}/regions/batch_box_state`,
    {
      method: 'POST',
      body: JSON.stringify({
        targets: targets.map((t) => ({ crop_id: t.cropId, box_id: t.boxId })),
        state,
        region_label_source: 'human',
      }),
    },
    signal,
  );
  return {
    updated: raw.updated,
    invalid: raw.invalid ?? [],
    conflicts: raw.conflicts ?? [],
    items: (raw.items ?? []).map(mapRawCrop),
  };
}

export interface RegionStatusEntry {
  value: string;
  label: string;
  role: string;
  terminal: boolean;
  human_writable: boolean;
  clears_box: boolean;
  wants_reason: boolean;
}

/** W8.7: per-box state styling/labels, distinct from the item-level
 *  `RegionStatusEntry` vocabulary above — a box's `state` is
 *  `proposed`/`accepted`/`rejected`/`false_positive`, never one of the
 *  item's `region_status` values. */
export type BoxStateTone = 'accepted' | 'proposed' | 'rejected' | 'neutral';

export interface BoxStateEntry {
  value: string;
  label: string;
  role: string;
  human_writable: boolean;
  exported: boolean;
  dashed: boolean;
  dim: boolean;
  badge: string | null;
  /**
   * PENDING_BACKEND_W8 (feat/w8-multibox-lockstep, docs/design/
   * w8-multibox-frontend-plan-2026-09-26.md): the backend approved this
   * as a follow-up to the W8.7 `box_states` vocabulary — the served
   * color/theme mapping for a box state, since `box_states` itself only
   * ever served `dashed`/`dim`/`badge` (styling flags, no color). Not in
   * the vendored OpenAPI snapshot yet (no `box_states` schema exists
   * there at all — `box_states` predates any contract-sync coverage);
   * remove this note (not widen it) once `npm run contract:sync` picks
   * it up. Optional so a pre-tone backend (or one that serves an
   * unrecognized value) renders neutral — see
   * `regionStatusesStore.boxStateTone()`.
   */
  tone?: BoxStateTone;
}

export interface RegionStatusesResponse {
  statuses: RegionStatusEntry[];
  confirm_status: string;
  reject_status: string;
  false_positive_status: string;
  /** W8.7: served box-state vocabulary (`GET /regions/statuses`), absent
   *  on a pre-W8 backend. */
  box_states?: BoxStateEntry[];
}

/** The deployment's region-status vocabulary (`GET {API_PREFIX}/regions/statuses`),
 *  meant to be loaded once by a store — see `$stores/regionStatuses.svelte`. */
export function getRegionStatuses(signal?: AbortSignal): Promise<RegionStatusesResponse> {
  return apiFetch<RegionStatusesResponse>(
    `${scoped()}${REGION_BASE}/statuses`,
    {},
    signal,
  );
}

/** What kind of writer a detector/source/chain-actor vocabulary entry names
 *  (`GET {API_PREFIX}/regions/vocabulary`, W0 naming-sweep finding m9). Forward-tolerant —
 *  an unrecognized role string is still carried through, it just falls
 *  back to the neutral chip color. */
export type RegionVocabularyRole =
  | 'detector'
  | 'segmenter'
  | 'ocr'
  | 'verifier'
  | 'human'
  | 'classifier'
  | 'proposal'
  | (string & {});

export interface RegionVocabularyEntry {
  id: string;
  label: string;
  role: RegionVocabularyRole;
  /** Only present on `detectors` entries — marks the values that can
   *  actually appear in stored `region_detector` (the detector filter's
   *  exact option list). */
  filterable?: boolean;
}

/** The active deployment's region-text validity rules (dq-region,
 *  2026-09-24) — `GET {API_PREFIX}/regions/vocabulary`'s `text_rules`, null
 *  without a region profile. Informational only: the frontend never
 *  re-implements these rules client-side, it just has somewhere to show
 *  them (e.g. next to a `region_text_vlm_invalid` badge). */
export interface RegionTextRules {
  uppercase: boolean;
  charset: string;
  len_min: number;
  len_max: number;
  format: string;
  reject_sequences: boolean;
  placeholders: string[];
  no_reading_words: string[];
  invalid_reasons: string[];
}

/** What kind of thing rejected a region candidate (OpenProcessor 3f1a11e,
 *  2026-09-24) — drives both the label lookup and the styling: a
 *  `model_verdict` is the verifier judging the box wrong, `automatic` is
 *  a geometry sanity gate, and `needs_human` means no verdict was given
 *  at all — it must never be worded as a rejection. */
export type RejectionReasonKind = 'model_verdict' | 'automatic' | 'needs_human';

/** One pipeline-written `region_rejection_reason` value
 *  (`GET {API_PREFIX}/regions/vocabulary`, OpenProcessor 3f1a11e). `match: 'exact'`
 *  entries match the stored value verbatim; `match: 'prefix'` entries
 *  match a stored-value prefix (e.g. `sanity_reject:`) with
 *  `label_template`'s `{detail}` filled from whatever follows the
 *  prefix. A stored value matching nothing here (older free-text human
 *  reasons) renders verbatim — see `regionVocabularyStore.rejectionReasonLabel`. */
export interface RejectionReasonEntry {
  id: string;
  label: string;
  kind: RejectionReasonKind;
  match: 'exact' | 'prefix';
  label_template: string | null;
}

export interface RegionVocabularyResponse {
  detectors: RegionVocabularyEntry[];
  region_sources: RegionVocabularyEntry[];
  chain_actors: RegionVocabularyEntry[];
  /** `region_text_choice` values (dq-region, 2026-09-24) — plain ids,
   *  e.g. `readers_agree` / `vlm_preferred` / `vlm_only` / `ocr_only` /
   *  `ocr_mode` / `vlm_invalid` / `no_valid_reading` / `human`. No
   *  served label yet; the UI titlecases the id as a placeholder (see
   *  `regionVocabularyStore.textChoiceLabel`). */
  text_choices: string[];
  /** The active profile's region-text validity rules, or `null` without
   *  a region profile. */
  text_rules: RegionTextRules | null;
  /** Labeled `region_rejection_reason` vocabulary (OpenProcessor 3f1a11e). */
  rejection_reasons: RejectionReasonEntry[];
  /** The active region profile (same value `/health` serves), or `null`
   *  when none is configured, in which case every list above is empty. */
  region_profile: ServedRegionProfile | null;
}

/** The deployment-configured detector/segmenter/verifier vocabulary
 *  (`GET {API_PREFIX}/regions/vocabulary`, W0 finding m9), built from the active region
 *  profile / ingest profiles / `OP_VLM_MODEL` — never a hardcoded model id.
 *  Meant to be loaded once by a store — see `$stores/regionVocabulary.svelte`. */
export async function getRegionVocabulary(
  signal?: AbortSignal,
): Promise<RegionVocabularyResponse> {
  const res = await apiFetch<Partial<RegionVocabularyResponse>>(
    `${scoped()}${REGION_BASE}/vocabulary`,
    {},
    signal,
  );
  return {
    detectors: res.detectors ?? [],
    region_sources: res.region_sources ?? [],
    chain_actors: res.chain_actors ?? [],
    text_choices: res.text_choices ?? [],
    text_rules: res.text_rules ?? null,
    rejection_reasons: res.rejection_reasons ?? [],
    region_profile: res.region_profile ?? null,
  };
}

/** One `value`/`label` option of a `ReviewFilterSpec` (3f1a11e adoption). */
export interface ReviewFilterOption {
  value: string;
  label: string;
}

/** Self-describing spec for one of a review tab's filters with a fixed
 *  value set (`GET {API_PREFIX}/review/tabs`, 3f1a11e adoption) — e.g. the
 *  region tab's `region_status` (all / detected only / verifier-rejected
 *  candidates only). `param` is the query parameter to send on both
 *  `GET {API_PREFIX}/review/{tab}` and its `/locate` route; an unknown value 400s.
 *  The frontend renders one `<select>` per entry generically — no
 *  tab-specific code reads `param` by name. */
export interface ReviewFilterSpec {
  param: string;
  kind: 'enum';
  label: string;
  options: ReviewFilterOption[];
}

/** One entry of `GET {API_PREFIX}/review/tabs` (W0 finding m9) — the served
 *  label/description for a review tab or preset id. */
export interface ReviewTabVocabularyEntry {
  id: string;
  label: string;
  description: string;
  /** Query parameters this tab honours — a parameter not listed is
   *  accepted and ignored server-side. Drives which filter-bar controls
   *  render for the active tab. */
  filters: string[];
  /** Values the tab applies when a filter is omitted, e.g.
   *  `{max_rank: 2}` for the two primary-subject tabs. */
  filter_defaults: Record<string, unknown>;
  /** Self-describing enum filters this tab honours (empty for none). */
  filter_specs: ReviewFilterSpec[];
}

/** `empty_state` on `GET {API_PREFIX}/review/tabs` — whether the
 *  deployment has ANY probe predictions or item scores at all, so an
 *  empty Uncertainty/Model-disagreements/score-sorted queue can point at
 *  the missing prerequisite (run a probe, compute scores) instead of just
 *  saying "empty". */
export interface ReviewEmptyState {
  has_probe_predictions: boolean;
  has_item_scores: boolean;
  /** Whether any labeled-dataset import has written labels (W10). */
  has_imported_labels: boolean;
}

/** `GET {API_PREFIX}/review/tabs`. */
export interface ReviewTabsResponse {
  tabs: ReviewTabVocabularyEntry[];
  empty_state: ReviewEmptyState;
}

/** Every review tab's served vocabulary, in served order, plus the
 *  deployment-wide `empty_state`. The frontend keeps its own tab
 *  structure/ids (`reviewTabs.ts`) and only overlays the served
 *  label/description/filters on top. */
export function getReviewTabs(signal?: AbortSignal): Promise<ReviewTabsResponse> {
  return apiFetch<ReviewTabsResponse>(`${scoped()}/review/tabs`, {}, signal);
}

/**
 * Fetch a single crop by id from the authoritative store. Used by the
 * review-page "Back" path so the operator sees what was actually
 * persisted rather than a possibly-stale local snapshot. Endpoint:
 * `GET {API_PREFIX}/crops/{crop_id}`.
 */
export async function getCrop(cropId: string, signal?: AbortSignal): Promise<Crop> {
  const raw = await apiFetch<RawCrop>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}`,
    {},
    signal,
  );
  return mapRawCrop(raw);
}

/**
 * PATCH a slot's item-level metadata (status / text / rejection reason)
 * without touching any box. The BODY KEYS are the spec's own wire field
 * names — for the region slot `region_status` / `region_rejection_reason`
 * only: its text is per box (no `text.valueField`), written with
 * `patchRegionBox`, and the backend rejects a `region_text` key. Keys
 * whose capability is absent, or whose value is `undefined` (as opposed
 * to `null`, which clears), are omitted.
 */
export async function patchSlotMeta(
  spec: SlotSpec,
  cropId: string,
  patch: {
    status?: string | null;
    text?: string | null;
    rejectionReason?: string | null;
  },
  signal?: AbortSignal,
): Promise<{ crop_id: string; updated_fields: string[]; item: Crop }> {
  const cap = spec.capabilities;
  const body: Record<string, unknown> = {};
  if (patch.status !== undefined && cap.lifecycle?.statusField) {
    body[cap.lifecycle.statusField] = patch.status;
  }
  if (patch.text !== undefined && cap.text?.valueField) {
    body[cap.text.valueField] = patch.text;
  }
  if (patch.rejectionReason !== undefined && cap.lifecycle?.rejectionReasonField) {
    body[cap.lifecycle.rejectionReasonField] = patch.rejectionReason;
  }
  const path = spec.endpoints.patchMeta?.(cropId);
  if (!path) {
    return Promise.reject(new Error(`slot "${spec.key}" has no patchMeta endpoint`));
  }
  const res = await apiFetch<{
    crop_id: string;
    updated_fields: string[];
    item: RawCrop;
  }>(`${scoped()}${path}`, { method: 'PATCH', body: JSON.stringify(body) }, signal);
  return { ...res, item: mapRawCrop(res.item) };
}

/**
 * Bulk-set region_status over many crops. Backend: `POST {API_PREFIX}/regions/batch_status`.
 * The cluster-view triage op: select outlier regions → mark all false_positive,
 * or bulk-confirm good regions (status='detected' + verified=true).
 *
 * Reads its path from `spec.endpoints.batchStatus` (Wave 2 C13,
 * docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §8.2 trap 2)
 * rather than hardcoding `${REGION_BASE}/batch_status` a second time —
 * closes the "declared but dead" gap without importing a specific
 * profile into this generic module (falls back to the REGION_BASE path
 * if a spec declares no batchStatus endpoint, matching today's only caller).
 */
export interface BatchStatusInvalidEntry {
  crop_id: string;
  detail: string;
}

export async function batchRegionStatus(
  spec: SlotSpec,
  cropIds: string[],
  status: string,
  opts: { verified?: boolean; labelSource?: string } = {},
  signal?: AbortSignal,
): Promise<{
  updated: number;
  // W8 (rev3, "one RegionBatchConflict shape across every region batch
  // route"): a pre-W8 backend serves only {crop_id, current_source}; a
  // W8 backend serves the full RegionBatchConflict. Typed as a superset
  // (every RegionBatchConflict field optional here) so both eras read
  // safely without a second type.
  conflicts: Array<{
    crop_id: string;
    current_source: string | null;
    error?: string;
    message?: string;
    current_region_revision?: number;
    current_box_ids?: string[];
    item?: RawCrop;
  }>;
  invalid: BatchStatusInvalidEntry[];
  items: RegionBrowseItem[];
}> {
  const path = spec.endpoints.batchStatus?.() ?? `${REGION_BASE}/batch_status`;
  const lc = spec.capabilities.lifecycle;
  if (!lc) {
    return Promise.reject(new Error(`slot "${spec.key}" has no lifecycle capability`));
  }
  const body: Record<string, unknown> = {
    crop_ids: cropIds,
    [lc.statusField]: status,
  };
  // p5 (2026-09-24 interactive pass): used to always send
  // `[verifiedField]: opts.verified ?? null`, so a bulk call that
  // never passed `verified` (e.g. reject/mark-false-positive) sent
  // an explicit `null` the server ignores. Omit the key entirely unless
  // the caller actually asked to set it.
  if (lc.verifiedField && opts.verified !== undefined) {
    body[lc.verifiedField] = opts.verified;
  }
  if (lc.labelSourceField) body[lc.labelSourceField] = opts.labelSource ?? 'human';
  return apiFetch(
    `${scoped()}${path}`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/**
 * Kick off a VLM-label run scoped to one cluster: `POST
 * {API_PREFIX}/vlm/label_cluster/{cluster_id}[?prompt_pack=&vlm=&acknowledge_external=]`. The server
 * selects every unvalidated, non-holdout, non-excluded member itself —
 * the frontend no longer fetches the crop page or chunks ids client-side
 * (that was the old `{API_PREFIX}/vlm/label_batch` chunk-of-64 loop,
 * deleted 2026-09-24 logic-moves W3). Returns the queued job's state
 * (`AutoLabelJobState`, defined below); 409 when another auto-label job
 * is already running. Progress and the final per-stage result come from
 * polling `{API_PREFIX}/pipeline/auto_label/status` — see
 * `pollAutoLabelJob`.
 */
export function runVlmOnCluster(
  clusterId: number,
  opts: {
    promptPack?: string | null;
    /** W9: a registry endpoint name (or `off`) for this run only. */
    vlm?: string | null;
    /** Sent only when `true`. */
    acknowledgeExternal?: boolean;
  } = {},
  signal?: AbortSignal,
): Promise<AutoLabelJobState> {
  return apiFetch<AutoLabelJobState>(
    `${scoped()}/vlm/label_cluster/${clusterId}${qs({
      prompt_pack: opts.promptPack ?? undefined,
      vlm: opts.vlm ?? undefined,
      acknowledge_external: opts.acknowledgeExternal === true ? true : undefined,
    })}`,
    { method: 'POST' },
    signal,
  );
}

/**
 * Poll a job until it leaves `running`, calling `onUpdate` with every
 * intermediate state so a caller can render stage/progress. Shared by
 * every caller of `runVlmOnCluster` (`/dashboard`, `/clusters/[id]`)
 * instead of each page hand-rolling its own `setTimeout` loop —
 * `AutoLabelPanel` keeps its own poller since it also needs to detect a
 * daemon-fired run while idle, which this helper, only ever started
 * right after `runVlmOnCluster`, does not.
 *
 * `expectedJobId` (M7, docs/design/interactive-pass-2026-09-24.md): when
 * given, polls `GET {API_PREFIX}/pipeline/auto_label/status/{job_id}`
 * (`getAutoLabelJobStatus`) for that exact job — the backend now serves
 * this per-job (b654da5), so there's no more race against
 * `{API_PREFIX}/pipeline/auto_label/status`'s single "current/most
 * recent job" slot answering with the *previous* job's already-terminal
 * status in the same tick a caller that just started a new one polls (a
 * real race, not hypothetical — observed live: a 1-crop cluster run
 * toasted "0 crops (0 updated)" because the read landed before the new
 * job had even flipped to `running`). A `404` (job not yet visible, or
 * never existed) is treated as "still waiting", bounded by `maxWaitMs` so
 * a backend that genuinely drops the job doesn't hang forever. Without
 * `expectedJobId`, falls back to `getAutoLabelStatus` (the "current job"
 * slot) unconditionally, same as before.
 */
export async function pollAutoLabelJob(
  onUpdate: (job: AutoLabelJobState) => void,
  signal?: AbortSignal,
  intervalMs = 1500,
  expectedJobId?: string,
  maxWaitMs = 5 * 60_000,
): Promise<AutoLabelJobState> {
  const deadline = Date.now() + maxWaitMs;
  for (;;) {
    const job =
      expectedJobId == null
        ? await getAutoLabelStatus(signal)
        : await getAutoLabelJobStatus(expectedJobId, signal);
    if (job != null) {
      onUpdate(job);
      if (job.status !== 'running') return job;
    } else if (Date.now() > deadline) {
      throw new Error(`Timed out waiting for job ${expectedJobId} to appear in status.`);
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

export type RefineClusterResponse = {
  cluster_id: number;
  n_members: number;
  n_subclusters: number;
  purity: number;
  subcluster_weighted_purity?: number;
  n_updated?: number;
  distance_threshold?: number;
  linkage?: string;
  metric?: string;
  action: 'refined' | 'skipped_too_small' | 'skipped_too_large';
  reason?: string;
};

export function refineCluster(
  clusterId: number,
  signal?: AbortSignal,
): Promise<RefineClusterResponse> {
  return apiFetch<RefineClusterResponse>(
    `${scoped()}/clusters/refine/${clusterId}`,
    { method: 'POST' },
    signal,
  );
}

export async function getReviewQueue(
  tab: ReviewTab,
  page = 1,
  pageSize = 30,
  filter: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<ReviewItem>> {
  // The {API_PREFIX}/review API ships bbox_norm + region_bbox_norm as
  // [x1,y1,x2,y2] arrays. The labeler's ReviewItem extends Crop where
  // bboxes are {cx,cy,w,h} objects. Normalize each item through
  // mapRawCrop so SlotBboxEditor + getThumbUrl + confirmSlot all see the
  // same shape regardless of the endpoint that produced the item.
  // proposed_class_id/name come straight off RawCrop/mapRawCrop now —
  // they're served on every crop-shaped item (item 11, 2026-09-24
  // logic-moves), not a review-only field — so only the genuinely
  // review-specific extras are declared here.
  type RawReviewItem = RawCrop & {
    reason?: string;
    probe_pred_class?: string | null;
    probe_pred_class_id?: number | null;
    probe_pred_entropy?: number | null;
    needs_new_class?: boolean;
    needs_new_class_note?: string | null;
  };
  type RawPage = {
    total: number;
    page: number;
    page_size: number;
    items: RawReviewItem[];
    /** Set when the requested `?sort=` fell back to the default — see
     *  PaginatedResponse.sort_fallback_reason in types.ts. */
    sort_fallback_reason?: string | null;
    /** The sort id actually applied — see PaginatedResponse.sort_applied. */
    sort_applied?: string | null;
    /** Set when total === 0 — see PaginatedResponse.empty_reason (#36 item 9). */
    empty_reason?: string | null;
  };
  const raw = await apiFetch<RawPage>(
    `${scoped()}/review/${tab}${qs({ page, page_size: pageSize, ...filter })}`,
    {},
    signal,
  );
  const items: ReviewItem[] = (raw.items ?? []).map((it) => {
    const base = mapRawCrop(it);
    return {
      ...base,
      reason: it.reason ?? '',
      probe_pred_class: it.probe_pred_class ?? null,
      probe_pred_class_id: it.probe_pred_class_id ?? null,
      probe_pred_entropy: it.probe_pred_entropy ?? null,
      needs_new_class: !!it.needs_new_class,
      needs_new_class_note: it.needs_new_class_note ?? null,
    };
  });
  return {
    items,
    total: raw.total ?? items.length,
    page: raw.page ?? page,
    page_size: raw.page_size ?? pageSize,
    sort_fallback_reason: raw.sort_fallback_reason ?? null,
    sort_applied: raw.sort_applied ?? null,
    empty_reason: raw.empty_reason ?? null,
  };
}

/** `GET {API_PREFIX}/review/{tab}/locate`— where a specific crop sits in a
 *  review queue under the given filters/sort, without paging through it
 *  by hand. Powers `/review?crop_id=` deep links (2026-09-24 logic-moves
 *  W5): `in_queue: false` means the crop doesn't match this tab's
 *  filters (or is already handled) — `reason` explains why when the
 *  backend sends one. */
export interface ReviewLocateResult {
  crop_id: string;
  in_queue: boolean;
  rank: number | null;
  page: number | null;
  page_size: number;
  total: number;
  reason: string | null;
  sort_applied: string | null;
  /** M11: mirrors `PaginatedResponse.sort_fallback_reason` — set when the
   *  sort `/locate` resolved (tab default, or the caller's own `sort`)
   *  fell back because its field has 0% coverage. */
  sort_fallback_reason: string | null;
}

export async function locateInReviewQueue(
  tab: string,
  cropId: string,
  pageSize: number,
  filter: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<ReviewLocateResult> {
  const raw = await apiFetch<Partial<ReviewLocateResult>>(
    `${scoped()}/review/${tab}/locate${qs({ crop_id: cropId, page_size: pageSize, ...filter })}`,
    {},
    signal,
  );
  return {
    crop_id: raw.crop_id ?? cropId,
    in_queue: !!raw.in_queue,
    rank: raw.rank ?? null,
    page: raw.page ?? null,
    page_size: raw.page_size ?? pageSize,
    total: raw.total ?? 0,
    reason: raw.reason ?? null,
    sort_applied: raw.sort_applied ?? null,
    sort_fallback_reason: raw.sort_fallback_reason ?? null,
  };
}

/** `GET {API_PREFIX}/review/new_class_proposals/summary` — the aggregate
 *  `/classes`'s Proposals section renders (top VLM-proposed-but-unmatched
 *  terms, with counts and a handful of sample crop ids each), distinct
 *  from paging the `new_class_proposals` review tab item-by-item. */
/** DQ-M11 fix (dq-queues cutover, 2026-09-24): `flag` is `null` for a
 *  term worth one-click creating (the only kind `top_terms` holds now),
 *  or the served reason it isn't: `existing_class` (map to `class_id`
 *  instead of creating), `generic_parent` (a super-category, e.g.
 *  a super-category term), or `non_object` (junk, e.g. "abstract_blur"). */
export interface NewClassProposalTerm {
  label: string;
  count: number;
  sample_crop_ids: string[];
  flag: 'existing_class' | 'generic_parent' | 'non_object' | null;
  class_id: number | null;
}

/** Server-side term-classification rules, rendered as help text —
 *  `generic_terms`/`non_object_terms` name the deployment's configured
 *  vocab (each overridable via its `_env` var). */
export interface NewClassTermRules {
  generic_terms: string[];
  non_object_terms: string[];
  registry_groups_are_generic: boolean;
  existing_classes_flagged: boolean;
  generic_terms_env: string;
  non_object_terms_env: string;
}

export interface NewClassProposalsSummary {
  total_pending: number;
  /** Queue items with no proposed term at all (a human "needs new
   *  class" flag with no name) — not represented in top_terms/
   *  flagged_terms, since there's no label to key a row on. */
  without_term: number;
  /** Only `flag: null` terms — every one is one-click "Create class &
   *  assign". */
  top_terms: NewClassProposalTerm[];
  /** The rest: `existing_class` / `generic_parent` / `non_object`. No
   *  create action is offered for these; `existing_class` offers
   *  map-to-`class_id` instead. */
  flagged_terms: NewClassProposalTerm[];
  term_rules: NewClassTermRules | null;
}

function asProposalTerms(v: unknown): NewClassProposalTerm[] {
  if (!Array.isArray(v)) return [];
  const out: NewClassProposalTerm[] = [];
  for (const raw of v) {
    if (typeof raw !== 'object' || raw === null) continue;
    const r = raw as Record<string, unknown>;
    if (typeof r.label !== 'string') continue;
    out.push({
      label: r.label,
      count: typeof r.count === 'number' ? r.count : 0,
      sample_crop_ids: Array.isArray(r.sample_crop_ids)
        ? (r.sample_crop_ids as string[])
        : [],
      flag:
        r.flag === 'existing_class' ||
        r.flag === 'generic_parent' ||
        r.flag === 'non_object'
          ? r.flag
          : null,
      class_id: typeof r.class_id === 'number' ? r.class_id : null,
    });
  }
  return out;
}

export async function getNewClassProposalsSummary(
  signal?: AbortSignal,
): Promise<NewClassProposalsSummary> {
  const raw = await apiFetch<Record<string, unknown>>(
    `${scoped()}/review/new_class_proposals/summary`,
    {},
    signal,
  );
  return {
    total_pending: typeof raw.total_pending === 'number' ? raw.total_pending : 0,
    without_term: typeof raw.without_term === 'number' ? raw.without_term : 0,
    top_terms: asProposalTerms(raw.top_terms),
    flagged_terms: asProposalTerms(raw.flagged_terms),
    term_rules:
      raw.term_rules && typeof raw.term_rules === 'object'
        ? (raw.term_rules as NewClassTermRules)
        : null,
  };
}

/**
 * `POST {API_PREFIX}/review/new_class_proposals/resolve` — bulk-resolve
 * every pending `vlm_new_class_pending` item proposing `body.label`, not
 * just the summary's capped `sample_crop_ids` preview. Exactly one of
 * `body.class_id` (map to an existing registry class) / `body.create`
 * (register a new one first) must be set — the backend 422s otherwise;
 * an unknown `class_id` is 400, a duplicate `create.class_name` is 409
 * (both surface as `ApiError.detail`).
 *
 * `dryRun: true` (`?dry_run=true`) reports `matched`/`matched_ids` and
 * writes/creates nothing — callers use it to show "this will assign N
 * crops" before the real resolve. Every write is a restorable
 * `class_id_history` snapshot server-side, so `undoLabelBatch` with the
 * response's `updated_ids` puts each item back exactly as a pending
 * proposal.
 */
export function resolveNewClassProposal(
  body: ResolveNewClassRequest,
  opts: { dryRun?: boolean } = {},
  signal?: AbortSignal,
): Promise<ResolveNewClassResponse> {
  return apiFetch<ResolveNewClassResponse>(
    `${scoped()}/review/new_class_proposals/resolve${qs({ dry_run: opts.dryRun || undefined })}`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

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

/**
 * Free-text semantic search over crops (P2-14). Backend:
 * `GET {API_PREFIX}/search/text`, gated behind the `semantic_search` overlay in
 * `{API_PREFIX}/methods` (see `isSemanticSearchAvailable` in `./strategies`) — a
 * caller must check that gate before rendering a UI that calls this.
 *
 * Mirrors `getReviewQueue`'s response-shape handling: the endpoint
 * returns `{items, total, page, page_size}` with items shaped like other
 * crop payloads plus a similarity score, normalized through `mapRawCrop`
 * so every existing crop-consuming component (CropCard, thumbnails,
 * label actions) keeps working unchanged against a search result.
 *
 * `filter` threads through arbitrary extra query params — same
 * `Record<string, unknown>` convention as `getReviewQueue` — so callers
 * can pass `cluster_id` (search scoped to one cluster on
 * `/clusters/[id]`) or the active review tab/filters (on `/review`)
 * without this function needing to know their shape.
 */
export async function searchCrops(
  q: string,
  page = 1,
  pageSize = 30,
  filter: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<SearchCrop>> {
  type RawSearchItem = RawCrop & { semantic_score: number | null };
  type RawPage = {
    total: number;
    page: number;
    page_size: number;
    items: RawSearchItem[];
  };
  const raw = await apiFetch<RawPage>(
    `${scoped()}/search/text${qs({ q, page, page_size: pageSize, ...filter })}`,
    {},
    signal,
  );
  const items: SearchCrop[] = (raw.items ?? []).map((it) => {
    const base = mapRawCrop(it);
    return {
      ...base,
      // The backend's `_hydrate_item` (OpenProcessor semantic_search.py) sends
      // the match score as `semantic_score`.
      similarity_score: it.semantic_score ?? 0,
    };
  });
  return {
    items,
    total: raw.total ?? items.length,
    page: raw.page ?? page,
    page_size: raw.page_size ?? pageSize,
  };
}

/**
 * Kick off a YOLO export job. Optional `version_tag` is included in the
 * manifest (Section L14). Server returns 202 + job id while it runs.
 */
export function exportYolo(
  opts: { version_tag?: string; require_fully_labeled_images?: boolean } = {},
  signal?: AbortSignal,
): Promise<ExportResult> {
  const body: Record<string, unknown> = {};
  if (opts.version_tag) body.version_tag = opts.version_tag;
  // Default is false server-side — only send it when the operator opted
  // in, so the payload stays minimal for the common case.
  if (opts.require_fully_labeled_images) {
    body.require_fully_labeled_images = true;
  }
  return apiFetch<ExportResult>(
    `${scoped()}/export/yolo`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Poll current export state. */
export function exportStatus(signal?: AbortSignal): Promise<ExportStatus> {
  return apiFetch<ExportStatus>(`${scoped()}/export/status`, {}, signal);
}

/** Options an operator picks per single-class export build. */
export interface SingleClassExportOptions {
  version_tag?: string;
  empty_bg_ratio?: number;
  max_positive_images?: number;
  skip_test_split?: boolean;
  dedup_threshold?: number | null;
  image_mode?: 'whole_frame' | 'item_crop';
  img_max_side?: 640 | 1280;
}

/**
 * Build a narrowed single-class dataset from a slot's declared export
 * (`DatasetExportSpec`). Synchronous on the server; returns the export
 * dir + counts when done. Only ever called behind
 * `isDatasetExportAvailable(…, spec.kind)` — see `/train`'s gate.
 */
export function exportSingleClass(
  spec: DatasetExportSpec,
  opts: SingleClassExportOptions = {},
  signal?: AbortSignal,
): Promise<SingleClassExportResult> {
  const body: Record<string, unknown> = {
    profile_name: spec.profileName,
    box_source: spec.boxSource,
    class_ids: spec.classIds,
  };
  if (spec.regionClassName) body.region_class_name = spec.regionClassName;
  for (const [k, v] of Object.entries(opts)) {
    if (v !== undefined) body[k] = v;
  }
  return apiFetch<SingleClassExportResult>(
    `${scoped()}${spec.buildPath}`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Last build of a slot's single-class export (its own `current` symlink). */
export function exportSingleClassStatus(
  spec: DatasetExportSpec,
  signal?: AbortSignal,
): Promise<SingleClassExportStatus> {
  return apiFetch<SingleClassExportStatus>(
    `${scoped()}${spec.statusPath}${qs({ profile_name: spec.profileName })}`,
    {},
    signal,
  );
}

/**
 * List every materialized dataset version on disk (newest first) so the
 * operator can train on any past export — a small sample, a larger subset, or
 * the full set — for consistent data re-use across model-size upgrades.
 */
export function listDatasets(
  filter: { kind?: string; profile_name?: string } = {},
  signal?: AbortSignal,
): Promise<ExportDatasetList> {
  return apiFetch<ExportDatasetList>(
    `${scoped()}/export/datasets${qs(filter)}`,
    {},
    signal,
  );
}

// -- classes mutators ----------------------------------------------------

export function addClass(
  payload: RegistryClassCreate,
  signal?: AbortSignal,
): Promise<{ class_id: number; class_name: string; group: string }> {
  return apiFetch<{ class_id: number; class_name: string; group: string }>(
    `${scoped()}/classes`,
    { method: 'POST', body: JSON.stringify(payload) },
    signal,
  );
}

export function renameClass(
  classId: number,
  payload: RegistryClassUpdate,
  signal?: AbortSignal,
): Promise<unknown> {
  return apiFetch<unknown>(
    `${scoped()}/classes/${classId}`,
    { method: 'PUT', body: JSON.stringify(payload) },
    signal,
  );
}

/**
 * The 409 `detail` `POST {API_PREFIX}/classes/{id}/deprecate` produces when
 * the class still has data referencing it (`{error:"class_still_referenced",
 * message, class_id, item_count, confirmed_label_count}`) — merge is the
 * only way to retire a class in that state, so the caller uses this to
 * offer the existing merge flow instead of a raw error toast.
 */
export interface ClassStillReferencedDetail {
  error: 'class_still_referenced';
  message: string;
  class_id: number;
  item_count: number;
  confirmed_label_count: number;
}

export function classStillReferencedDetail(
  e: unknown,
): ClassStillReferencedDetail | null {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (d.error !== 'class_still_referenced') return null;
  if (typeof d.message !== 'string') return null;
  if (typeof d.class_id !== 'number') return null;
  if (typeof d.item_count !== 'number' || typeof d.confirmed_label_count !== 'number') {
    return null;
  }
  return {
    error: 'class_still_referenced',
    message: d.message,
    class_id: d.class_id,
    item_count: d.item_count,
    confirmed_label_count: d.confirmed_label_count,
  };
}

/** `POST /classes/{id}/restore`'s structured 409 for a class that was
 *  merged into another (OpenProcessor 4c125ec, F-56). */
export interface ClassMergedDetail {
  error: 'class_merged';
  message: string;
  class_id: number;
  merged_into: { class_id: number; class_name: string };
  hint: string | null;
}

export function classMergedDetail(e: unknown): ClassMergedDetail | null {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object') return null;
  const d = detail as Record<string, unknown>;
  if (d.error !== 'class_merged' || typeof d.message !== 'string') return null;
  const into = d.merged_into as Record<string, unknown> | null | undefined;
  if (!into || typeof into.class_id !== 'number') return null;
  return {
    error: 'class_merged',
    message: d.message,
    class_id: typeof d.class_id === 'number' ? d.class_id : -1,
    merged_into: {
      class_id: into.class_id,
      class_name:
        typeof into.class_name === 'string' ? into.class_name : String(into.class_id),
    },
    hint: typeof d.hint === 'string' ? d.hint : null,
  };
}

/** The operator-facing text for a merged-class restore refusal: the
 *  merge target by its served name, then the server's own message and
 *  hint verbatim. */
export function classMergedRestoreText(d: ClassMergedDetail): string {
  return [
    `Merged into ${d.merged_into.class_name}; un-merge isn't supported.`,
    d.message,
    d.hint ?? '',
  ]
    .filter((s) => s.length > 0)
    .join(' ');
}

/**
 * Retire a class with no data yet — no merge target needed. Idempotent
 * (calling on an already-deprecated class just returns it) and clears any
 * bound `hotkey_letter` server-side. 404 for an unknown id; 409 (see
 * `classStillReferencedDetail`) while any item/confirmed-label doc still
 * carries this `class_id` — merge instead. The response is the backend's
 * own `RegistryClassEntry` shape (`class_id`/`class_name`/…, not the
 * labeler's `RegistryClass`); callers refetch `classesStore` rather than
 * mapping it.
 */
export function deprecateClass(classId: number, signal?: AbortSignal): Promise<unknown> {
  return apiFetch<unknown>(
    `${scoped()}/classes/${classId}/deprecate`,
    { method: 'POST' },
    signal,
  );
}

/**
 * Undo `deprecateClass`. 404 for an unknown id; 409 with a PLAIN STRING
 * `detail` (not the structured shape above) when a live class already
 * uses this class's name — show `ApiError.detail` verbatim.
 */
export function restoreClass(classId: number, signal?: AbortSignal): Promise<unknown> {
  return apiFetch<unknown>(
    `${scoped()}/classes/${classId}/restore`,
    { method: 'POST' },
    signal,
  );
}

export function mergeClasses(
  payload: RegistryClassMerge,
  signal?: AbortSignal,
): Promise<{
  source_id: number;
  target_id: number;
  deprecated: boolean;
  source_name: string;
  target_name: string;
}> {
  // The backend does not return a relabeled count -- don't claim one.
  return apiFetch<{
    source_id: number;
    target_id: number;
    deprecated: boolean;
    source_name: string;
    target_name: string;
  }>(
    `${scoped()}/classes/merge`,
    { method: 'POST', body: JSON.stringify(payload) },
    signal,
  );
}

/**
 * `POST {API_PREFIX}/classes/merge?dry_run=true` — reports what a real merge
 * would do (`would_relabel`, `validations_carried_over`, `holdout_blocking`,
 * `blocked`) and writes nothing. The merge dialog calls this before every
 * real merge so the operator sees the blast radius first.
 */
export function previewClassMerge(
  payload: RegistryClassMerge,
  signal?: AbortSignal,
): Promise<ClassMergeDryRun> {
  return apiFetch<ClassMergeDryRun>(
    `${scoped()}/classes/merge${qs({ dry_run: true })}`,
    { method: 'POST', body: JSON.stringify(payload) },
    signal,
  );
}

/** What kind of writer a `class_source` value names. */
export type ClassSourceRole =
  | 'proposal'
  | 'low_conf'
  | 'model'
  | 'vlm'
  | 'vlm_unmatched'
  | 'vlm_new_class_pending'
  | 'vlm_reclassified'
  | 'cluster'
  | 'human'
  | 'merge'
  | 'label_import'
  | (string & {});

/** One `class_source` value this deployment can write. */
export interface ClassSource {
  id: string;
  label: string;
  role: ClassSourceRole;
}

/** `GET {API_PREFIX}/class_sources` — every `class_source` value this
 *  deployment can write, including the ingest detectors' config-derived
 *  ones (`{primary}_proposal`, …). */
export async function getClassSources(signal?: AbortSignal): Promise<ClassSource[]> {
  const res = await apiFetch<{ class_sources?: ClassSource[] }>(
    `${scoped()}/class_sources`,
    {},
    signal,
  );
  return (res.class_sources ?? []).filter(
    (c) => typeof c?.id === 'string' && c.id.length > 0 && typeof c.label === 'string',
  );
}

export function syncClassesToOpensearch(
  signal?: AbortSignal,
): Promise<{ upserted: number; n_classes: number }> {
  return apiFetch<{ upserted: number; n_classes: number }>(
    `${scoped()}/classes/sync_to_opensearch`,
    { method: 'POST' },
    signal,
  );
}

// -- DnD: move crops between clusters ------------------------------------

export function moveCropsToCluster(
  cropIds: string[],
  targetClusterId: number,
  signal?: AbortSignal,
): Promise<BulkLabelResult> {
  // Backend returns the same {updated, conflicts: [...]} shape as
  // {API_PREFIX}/crops/batch_label. Reuse the type so both call sites share the
  // conflict-handling code path.
  return apiFetch<BulkLabelResult>(
    `${scoped()}/crops/move`,
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds, cluster_id: targetClusterId }),
    },
    signal,
  );
}

// -- exclude / ignore crops ----------------------------------------------

/** Reasons the operator can attach when ignoring crops. 'ignore' is the
 *  default one-click value; the others record *why* for later analysis. */
export type ExcludeReason =
  'ignore' | 'blurry' | 'unidentifiable' | 'not_the_subject' | 'partial_crop';

export function excludeCrops(
  cropIds: string[],
  reason: ExcludeReason = 'ignore',
  signal?: AbortSignal,
): Promise<{ excluded: number; errors: number }> {
  return apiFetch<{ excluded: number; errors: number }>(
    `${scoped()}/crops/batch_exclude`,
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds, reason }),
    },
    signal,
  );
}

export function unexcludeCrops(
  cropIds: string[],
  signal?: AbortSignal,
): Promise<{ unexcluded: number; errors: number }> {
  return apiFetch<{ unexcluded: number; errors: number }>(
    `${scoped()}/crops/batch_unexclude`,
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds }),
    },
    signal,
  );
}

// -- needs new class flag ------------------------------------------------

export function flagNeedsNewClass(
  cropIds: string[],
  note: string = '',
  signal?: AbortSignal,
): Promise<{ flagged: number; errors: number }> {
  return apiFetch<{ flagged: number; errors: number }>(
    `${scoped()}/crops/flag_new_class`,
    {
      method: 'POST',
      body: JSON.stringify({ crop_ids: cropIds, note }),
    },
    signal,
  );
}

// -- test holdout --------------------------------------------------------

/**
 * `POST {API_PREFIX}/test_holdout/freeze` — deterministic (SHA1-of-
 * `crop_id` per class), so the body is `{percent}` only as of
 * OpenProcessor df01309; an extra field such as `seed` is a 422
 * (`additionalProperties: false`). `force` re-runs an existing freeze
 * and is a query param, not a body field.
 */
export function freezeTestHoldout(
  payload: { percent: number; force?: boolean },
  signal?: AbortSignal,
): Promise<TestHoldoutFreezeResult> {
  return apiFetch<TestHoldoutFreezeResult>(
    `${scoped()}/test_holdout/freeze${qs({ force: payload.force ? true : undefined })}`,
    { method: 'POST', body: JSON.stringify({ percent: payload.percent }) },
    signal,
  );
}

export function getTestHoldoutStats(signal?: AbortSignal): Promise<TestHoldoutStats> {
  return apiFetch<TestHoldoutStats>(`${scoped()}/test_holdout/stats`, {}, signal);
}

// -- registry/manifest downloads (used as anchor `download` URLs) --------

export function getClassRegistryUrl(): string {
  return `${apiBase}${scoped()}/export/registry/class_registry.json`;
}

export function getDataYamlUrl(): string {
  return `${apiBase}${scoped()}/export/registry/data.yaml`;
}

export function getManifestUrl(): string {
  return `${apiBase}${scoped()}/export/registry/manifest.json`;
}

// -- image URL helpers (no fetch — used directly in <img src=...>) -------

/**
 * URL for a crop thumbnail. Defaults to 160px — small enough to load
 * fast for fast grid scanning of thousands of crops, large enough that
 * subject details (color, shape, fine texture) remain readable.
 * Server-side aspect-correct rendering preserves the bbox proportions.
 *
 * Pass a larger ``size`` (256-512) for click-to-inspect / focused review
 * where rendering quality matters more than transfer speed.
 */
export function getThumbUrl(cropId: string, size: number = 160): string {
  return `${apiBase}${scoped()}/crops/${encodeURIComponent(cropId)}/thumbnail?size=${size}`;
}

export function getSourceImageUrl(cropId: string): string {
  return `${apiBase}${scoped()}/crops/${encodeURIComponent(cropId)}/image`;
}

/**
 * Source image downscaled to ~`maxDim`px on the longest side — plenty
 * for `SourceImageOverlay`'s client-drawn boxes to be readable without
 * pushing 2+ MB per open. K6 (docs/design/
 * k6-frontend-overlay-plan-2026-09-24.md): the backend no longer burns
 * any box/label into this image (`getSourceImageWithBbox` — the
 * server-overlay-era name — is retired; `SourceImageOverlay` draws
 * everything itself from `getCropContext`). Callers that need a
 * pixel-accurate frame (e.g. SlotBboxEditor) should hit
 * ``getSourceImageFull`` instead.
 */
export function getSourceImageScaled(cropId: string, maxDim: number = 1280): string {
  return `${apiBase}${scoped()}/crops/${encodeURIComponent(cropId)}/image?max_dim=${maxDim}`;
}

/** Full-resolution source image; used by SlotBboxEditor where pixel accuracy matters. */
export function getSourceImageFull(cropId: string): string {
  return `${apiBase}${scoped()}/crops/${encodeURIComponent(cropId)}/image`;
}

// -- training endpoints --------------------------------------------------
//
// Mirror the FastAPI `{API_PREFIX}/train/*` router. The labeler `/train` page is
// the only consumer; types live in `./types_train.ts` so the existing
// types.ts stays focused on the labeling data model.

/**
 * Run the preflight checks for a candidate spec WITHOUT writing
 * `job.json`. Used for inline form validation.
 *
 * The endpoint never throws on blocking issues — it returns
 * `{blocked, checks, summary}` so the UI can render every row.
 */
export function trainPreflight(
  spec: TrainJobSpec,
  signal?: AbortSignal,
): Promise<PreflightReport> {
  return apiFetch<PreflightReport>(
    `${scoped()}/train/preflight`,
    { method: 'POST', body: JSON.stringify(spec) },
    signal,
  );
}

/**
 * Start a single training run. Backend returns 422 with the preflight
 * report on blocking checks unless `force=true`. We surface those to
 * the caller as ApiError; the form reads `body.preflight` and renders.
 */
export function trainStart(
  spec: TrainJobSpec,
  force: boolean = false,
  signal?: AbortSignal,
): Promise<StartTrainResponse> {
  return apiFetch<StartTrainResponse>(
    `${scoped()}/train/start${qs({ force: force ? true : undefined })}`,
    { method: 'POST', body: JSON.stringify(spec) },
    signal,
  );
}

/** Submit a multi-size training campaign (queued back-to-back). */
export function trainStartCampaign(
  spec: TrainCampaignSpec,
  force: boolean = false,
  signal?: AbortSignal,
): Promise<StartCampaignResponse> {
  return apiFetch<StartCampaignResponse>(
    `${scoped()}/train/start_campaign${qs({ force: force ? true : undefined })}`,
    { method: 'POST', body: JSON.stringify(spec) },
    signal,
  );
}

/**
 * Most-recent or active job's `status.json`. Returns null when there's
 * never been a run.
 *
 * Pass a `jobId` to fetch a specific run (404 on missing).
 */
export async function getTrainStatus(
  jobId?: string,
  signal?: AbortSignal,
): Promise<TrainJobStatus | null> {
  const path = jobId
    ? `${scoped()}/train/status/${encodeURIComponent(jobId)}`
    : `${scoped()}/train/status`;
  try {
    return await apiFetch<TrainJobStatus | null>(path, {}, signal);
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return null;
    throw e;
  }
}

export function getTrainRuns(
  limit: number = 50,
  offset: number = 0,
  signal?: AbortSignal,
): Promise<RunsListResponse> {
  return apiFetch<RunsListResponse>(
    `${scoped()}/train/runs${qs({ limit, offset })}`,
    {},
    signal,
  );
}

export function tailTrainLog(
  jobId: string,
  lines: number = 200,
  signal?: AbortSignal,
): Promise<LogTailResponse> {
  return apiFetch<LogTailResponse>(
    `${scoped()}/train/log/tail/${encodeURIComponent(jobId)}${qs({ lines })}`,
    {},
    signal,
  );
}

export function cancelTrainJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<CancelResponse> {
  return apiFetch<CancelResponse>(
    `${scoped()}/train/cancel/${encodeURIComponent(jobId)}`,
    { method: 'POST' },
    signal,
  );
}

export function cancelTrainCampaign(
  campaignId: string,
  signal?: AbortSignal,
): Promise<CancelResponse> {
  return apiFetch<CancelResponse>(
    `${scoped()}/train/cancel_campaign/${encodeURIComponent(campaignId)}`,
    { method: 'POST' },
    signal,
  );
}

export function getTrainProfiles(signal?: AbortSignal): Promise<ProfilesResponse> {
  return apiFetch<ProfilesResponse>(`${scoped()}/train/profiles`, {}, signal);
}

/** One GPU claim the backend allows for a training run. */
export interface TrainGpuOption {
  /** The exact `cuda_visible_devices` value to send. */
  value: string;
  gpu_ids: number[];
  label: string;
  /** What the claim does to other services (e.g. which containers stop). */
  advisory: string | null;
  stops_containers: string[];
  default: boolean;
}

/** `GET {API_PREFIX}/train/gpus`. `unrestricted` means no allowlist is
 *  configured, so any `cuda_visible_devices` value may be entered. */
export interface TrainGpuOptionsResponse {
  options: TrainGpuOption[];
  allowed_ids: number[];
  unrestricted: boolean;
}

export function getTrainGpus(signal?: AbortSignal): Promise<TrainGpuOptionsResponse> {
  return apiFetch<TrainGpuOptionsResponse>(`${scoped()}/train/gpus`, {}, signal);
}

/** The served default claim, or '' when none is marked (the backend then
 *  picks from its allowlist when `cuda_visible_devices` is omitted). */
export function defaultGpuValue(res: TrainGpuOptionsResponse): string {
  return res.options.find((o) => o.default)?.value ?? '';
}

export function getTrainPresets(signal?: AbortSignal): Promise<PresetsResponse> {
  return apiFetch<PresetsResponse>(`${scoped()}/train/presets`, {}, signal);
}

/**
 * `GET {API_PREFIX}/train/augmentation_presets` (OpenProcessor df01309) —
 * the trainer's real preset catalog, for `AugmentationPanel`'s picker.
 */
export function getAugmentationPresets(
  signal?: AbortSignal,
): Promise<AugmentationPresetsResponse> {
  return apiFetch<AugmentationPresetsResponse>(
    `${scoped()}/train/augmentation_presets`,
    {},
    signal,
  );
}

export function promoteTrainJob(
  jobId: string,
  body: PromoteRequest,
  signal?: AbortSignal,
): Promise<PromoteResponse> {
  return apiFetch<PromoteResponse>(
    `${scoped()}/train/promote/${encodeURIComponent(jobId)}`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/**
 * Run manifest (design §15.4) — full lineage envelope: dataset SHA,
 * class remap, code versions, eval results. 404 means the run finished
 * before the manifest writer was added.
 */
export function getTrainManifest(
  jobId: string,
  signal?: AbortSignal,
): Promise<TrainManifest> {
  return apiFetch<TrainManifest>(
    `${scoped()}/train/manifest/${encodeURIComponent(jobId)}`,
    {},
    signal,
  );
}

// -- Probe (#36 item 8) ---------------------------------------------------
// Wraps POST {API_PREFIX}/probe/{run,cancel} / GET {API_PREFIX}/probe/status —
// starts a probe pass from a finished training run's export, populating
// probe_pred_* on items (the Uncertainty/Model-disagreements queues' one
// prerequisite, per item 9's empty_state). Same idempotent job-poll shape
// as scores/embedding-viz jobs: one job at a time, GET /probe/status is
// the single source of truth for what's running.

export interface ProbeRunRequest {
  job_id: string;
  architecture?: string;
  gpu?: string | null;
  resume?: boolean;
}

/** `ProbeStatusResponse` — served verbatim, including `error`, which is
 *  the backend's own message (a GPU-arbiter claim failure, a training job
 *  that isn't finished / has no exported checkpoint, etc.) and is always
 *  rendered as-is, never reworded. */
export interface ProbeStatusResponse {
  status: string;
  job_id?: string | null;
  train_job_id?: string | null;
  gpu?: string | null;
  model_path?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  updated_count?: number | null;
  error?: string | null;
  /** Read-only echo of `OP_PROBE_ACTIONABLE_MIN_CONFIDENCE` (default 0.5)
   *  — the threshold the server applies to compute the item wire's
   *  `probe_actionable`. No client code recomputes actionability against
   *  this; it exists purely to render "threshold: NN%" where a probe
   *  status is already in hand. */
  actionable_min_confidence?: number | null;
}

/** Start a probe pass from `trainJobId`'s finished export. 409 when the
 *  training job isn't finished / has no exported checkpoint, or a probe
 *  is already running — surfaced via `ApiError.detail`, never guessed. */
export function runProbe(
  trainJobId: string,
  opts: { architecture?: string; gpu?: string | null; resume?: boolean } = {},
  signal?: AbortSignal,
): Promise<ProbeStatusResponse> {
  const body: ProbeRunRequest = { job_id: trainJobId, ...opts };
  return apiFetch<ProbeStatusResponse>(
    `${scoped()}/probe/run`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Poll the current/last probe job. */
export function getProbeStatus(signal?: AbortSignal): Promise<ProbeStatusResponse> {
  return apiFetch<ProbeStatusResponse>(`${scoped()}/probe/status`, {}, signal);
}

/** Best-effort cancel of the active probe job. */
export function cancelProbe(signal?: AbortSignal): Promise<ProbeStatusResponse> {
  return apiFetch<ProbeStatusResponse>(
    `${scoped()}/probe/cancel`,
    { method: 'POST' },
    signal,
  );
}

// -- Auto-label (recluster) job ------------------------------------------
// Wraps POST {API_PREFIX}/pipeline/auto_label/{start,status,cancel}. The pipeline
// re-runs prototype assignment → cluster_id normalize → AHC residuals → auto-
// promote → VLM sweep, fixing prototype drift and stale cluster_id on
// labeled crops. Hours at HDD scale; the panel polls status while it runs.

export interface AutoLabelStartParams {
  train_clusters?: boolean;
  promote_min_purity?: number;
  promote_min_members?: number;
  vlm_batch_size?: number;
  vlm_concurrency?: number;
  max_vlm_crops?: number;
  classifier_confidence_skip_vlm?: number;
  /** When true, broaden the residual AHC pool to include items already
   *  in candidate clusters so smaller candidates can merge into bigger
   *  ones. Default false — only fresh / class-bucketed items are
   *  re-clustered. */
  recluster_unvalidated?: boolean;
  // -- Cluster scope (primary-subject gate) ------------------------------
  /** Train + assign only crops with crop_rank_in_image <= this (1 = largest,
   *  2 = largest + 2nd). Smaller crops are parked. Needs full rank/blur
   *  backfill; the run blocks otherwise. */
  gate_max_rank?: number | null;
  /** Train + assign only crops with blur_lap_ratio >= this. */
  gate_min_blur_ratio?: number | null;
  /** IVF fixed centroid count (default 512). Sweep down with the gate on. */
  n_clusters?: number | null;
  // -- VLM-assisted scoping (2026-09-20 contract, this plan §1.3) --------
  /**
   * Scope the run to a single class instead of the whole pool — "just
   * help me with forklifts right now". Same `class_id` filter convention
   * as `getCrops`/`getClusters`/`/select/diverse` elsewhere in this
   * file. `null`/omitted = today's unscoped, whole-dataset behavior, and
   * `qs()` drops it entirely so an unscoped request stays byte-identical
   * to every request this app has ever sent.
   *
   * Accepted by OpenProcessor `main`'s `auto_label/start`. The UI never
   * sends it unless `/methods` advertises the assist axes — see
   * `isScopedAssistAvailable` in `$lib/strategies`.
   */
  class_id?: number | null;
  /**
   * Per-run override of the settings-doc prompt-pack default, for this
   * job only — omitted/null means the deployment default. An unknown id
   * is a 422 (`unknownStrategyDetail`); the resolved value is echoed in
   * the job state's `args`. Produced in exactly one place
   * (`createAssistScope().toStartParams()` in `$lib/assistScope.svelte`).
   */
  prompt_pack?: string | null;
  /**
   * W9: per-run VLM endpoint (a registry name, or `off`) for this job
   * only; omitted/null = the project's active endpoint. An unknown id is
   * a 422 `unknown_vlm`. Produced in exactly one place
   * (`createAssistScope().toStartParams()`).
   */
  vlm?: string | null;
  /**
   * W9: acknowledges that this run's crops go to an external endpoint.
   * Sent only when `true` (`startAutoLabel` drops `false`).
   */
  acknowledge_external?: boolean;
  /**
   * G5: `pipeline.py`'s `run_vlm: bool = Query(False)` — the VLM sweep
   * stage is opt-in server-side and defaults off. Previously never sent
   * at all, so a scoped run (a class or pack picked via AssistScopeBar)
   * silently skipped the VLM stage it claimed to scope
   * (`result.stages.vlm.skipped=true` live). Omitted/undefined keeps the
   * request byte-identical to an unscoped run; the caller decides when
   * to send `true`.
   */
  run_vlm?: boolean;
}

export type AutoLabelStatus = 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface AutoLabelJobState {
  job_id: string;
  status: AutoLabelStatus;
  stage: string;
  processed: number;
  total: number;
  started_at: number;
  finished_at: number;
  error: string | null;
  result: Record<string, unknown>;
  args: Record<string, unknown>;
  eta_seconds: number | null;
  elapsed_seconds: number;
  // GPU clustering telemetry (optional — older worker versions omit).
  // `backend` is 'gpu' when cluster_residuals' UMAP ran on cuML;
  // 'cpu' for sklearn fallback. `backend_detail` is a human-readable
  // chip ("gpu (cuml 26.4 · A6000 GPU 0 · 9824/49152 MB free · nn_descent)"
  // or "cpu (sklearn 1.4 · umap-learn 0.5)").
  backend?: 'gpu' | 'cpu' | null;
  backend_detail?: string | null;
  free_vram_mb?: number | null;
  peak_vram_mb?: number | null;
  stage_durations?: Record<string, number>;
}

export function startAutoLabel(
  params: AutoLabelStartParams = {},
  signal?: AbortSignal,
): Promise<AutoLabelJobState> {
  const { acknowledge_external: ack, ...rest } = params;
  const sent = ack === true ? { ...rest, acknowledge_external: true } : rest;
  return apiFetch<AutoLabelJobState>(
    `${scoped()}/pipeline/auto_label/start${qs(sent as Record<string, unknown>)}`,
    { method: 'POST' },
    signal,
  );
}

export function getAutoLabelStatus(signal?: AbortSignal): Promise<AutoLabelJobState> {
  return apiFetch<AutoLabelJobState>(
    `${scoped()}/pipeline/auto_label/status`,
    {},
    signal,
  );
}

/**
 * `GET {API_PREFIX}/pipeline/auto_label/status/{job_id}` (M7,
 * docs/design/interactive-pass-2026-09-24.md): poll the exact job a caller
 * started, instead of the single "current/most recent job" slot
 * `getAutoLabelStatus` reads. Returns `null` for a `404` — an id no job
 * ever had (or not a 32-hex id) — so a caller can distinguish "not there
 * yet / never existed" from a real fetch failure without inspecting
 * `ApiError.status` itself.
 */
export async function getAutoLabelJobStatus(
  jobId: string,
  signal?: AbortSignal,
): Promise<AutoLabelJobState | null> {
  try {
    return await apiFetch<AutoLabelJobState>(
      `${scoped()}/pipeline/auto_label/status/${encodeURIComponent(jobId)}`,
      {},
      signal,
    );
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return null;
    throw e;
  }
}

export function cancelAutoLabel(
  signal?: AbortSignal,
): Promise<AutoLabelJobState & { cancelled: boolean }> {
  return apiFetch<AutoLabelJobState & { cancelled: boolean }>(
    `${scoped()}/pipeline/auto_label/cancel`,
    { method: 'POST' },
    signal,
  );
}

// ===========================================================================
// Ingest ({API_PREFIX}/ingest/*) — bring images into the pool.
// docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §B.2.

export function getIngestStatus(signal?: AbortSignal): Promise<IngestStatus> {
  return apiFetch<IngestStatus>(`${scoped()}/ingest/status`, {}, signal);
}

export function getRegionDrain(signal?: AbortSignal): Promise<RegionDrain> {
  return apiFetch<RegionDrain>(`${scoped()}/ingest/region_drain`, {}, signal);
}

export function ingestPathLookup(
  ids: string[],
  signal?: AbortSignal,
): Promise<IngestPathLookupResponse> {
  return apiFetch<IngestPathLookupResponse>(
    `${scoped()}/ingest/path_lookup`,
    { method: 'POST', body: JSON.stringify({ image_paths: ids }) },
    signal,
  );
}

/**
 * `POST {API_PREFIX}/ingest/upload` — multipart. The browser must set its
 * own `Content-Type` boundary; see the `apiFetch` FormData carve-out above.
 */
export function ingestUpload(
  req: IngestUploadRequest,
  signal?: AbortSignal,
): Promise<BatchIngestResponse> {
  const fd = new FormData();
  for (const f of req.files) fd.append('images', f, f.name);
  fd.append('image_paths', JSON.stringify(req.identifiers));
  fd.append('source', req.source);
  return apiFetch<BatchIngestResponse>(
    `${scoped()}/ingest/upload`,
    { method: 'POST', body: fd },
    signal,
  );
}

export async function ingestBatch(
  req: IngestBatchRequest,
  signal?: AbortSignal,
): Promise<BatchIngestResponse> {
  assertNonEmptyBatch('ingest', req.items);
  return apiFetch<BatchIngestResponse>(
    `${scoped()}/ingest/batch`,
    { method: 'POST', body: JSON.stringify(req) },
    signal,
  );
}

/**
 * Typed ingest capability + limits, actually enforced by
 * `/ingest/upload`/`/ingest/batch`/`/ingest/region_drain`.
 */
export function getIngestConfig(signal?: AbortSignal): Promise<IngestConfig> {
  return apiFetch<IngestConfig>(`${scoped()}/ingest/config`, {}, signal);
}

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

// -- labeled-dataset import and Reprocess (OpenProcessor W10) ------------
//
// any_domain_plan.md §7.12 / W10.14; docs/design/
// w10-import-reprocess-ui-plan-2026-09-27.md. Every route is scoped to the
// active project (the import's target is the path's project; no body
// carries `project`). A backend without W10 404s `GET /datasets/formats`,
// which `datasetsAvailability` treats as "not deployed yet".

export function getDatasetFormats(signal?: AbortSignal): Promise<DatasetFormatsResponse> {
  return apiFetch<DatasetFormatsResponse>(`${scoped()}/datasets/formats`, {}, signal);
}

/** `POST /datasets/uploads` — one multipart `file` (a .zip/.tar/.tar.gz),
 *  streamed server-side; the response's `dataset_path` is what the
 *  preview then reads. */
export function uploadDatasetArchive(
  file: File,
  signal?: AbortSignal,
): Promise<DatasetUploadResponse> {
  const fd = new FormData();
  fd.append('file', file, file.name);
  return apiFetch<DatasetUploadResponse>(
    `${scoped()}/datasets/uploads`,
    { method: 'POST', body: fd },
    signal,
  );
}

/** Dry run: writes nothing. Dataset problems come back as `issues`,
 *  never as a 4xx (only a malformed body or a disallowed root 422s). */
export function previewDataset(
  body: DatasetPreviewRequest,
  signal?: AbortSignal,
): Promise<DatasetPreview> {
  return apiFetch<DatasetPreview>(
    `${scoped()}/datasets/preview`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** 202 with a new job, or 200 with the existing one (`reused: true`). */
export function startDatasetImport(
  body: DatasetImportRequest,
  signal?: AbortSignal,
): Promise<DatasetImportJob> {
  return apiFetch<DatasetImportJob>(
    `${scoped()}/datasets/imports`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

export function listDatasetImports(
  params: { page?: number; page_size?: number; status?: string | null } = {},
  signal?: AbortSignal,
): Promise<DatasetImportList> {
  return apiFetch<DatasetImportList>(
    `${scoped()}/datasets/imports${qs({
      page: params.page,
      page_size: params.page_size,
      status: params.status,
    })}`,
    {},
    signal,
  );
}

export function getDatasetImport(
  importId: string,
  signal?: AbortSignal,
): Promise<DatasetImportJob> {
  return apiFetch<DatasetImportJob>(
    `${scoped()}/datasets/imports/${encodeURIComponent(importId)}`,
    {},
    signal,
  );
}

export function getDatasetImportIssues(
  importId: string,
  params: { code?: string | null; page?: number; page_size?: number } = {},
  signal?: AbortSignal,
): Promise<DatasetIssuePage> {
  return apiFetch<DatasetIssuePage>(
    `${scoped()}/datasets/imports/${encodeURIComponent(importId)}/issues${qs({
      code: params.code,
      page: params.page,
      page_size: params.page_size,
    })}`,
    {},
    signal,
  );
}

export function getDatasetImportEntries(
  importId: string,
  params: {
    split?: string | null;
    label_state?: string | null;
    status?: string | null;
    page?: number;
    page_size?: number;
  } = {},
  signal?: AbortSignal,
): Promise<DatasetImportEntryPage> {
  return apiFetch<DatasetImportEntryPage>(
    `${scoped()}/datasets/imports/${encodeURIComponent(importId)}/entries${qs({
      split: params.split,
      label_state: params.label_state,
      status: params.status,
      page: params.page,
      page_size: params.page_size,
    })}`,
    {},
    signal,
  );
}

export function cancelDatasetImport(
  importId: string,
  signal?: AbortSignal,
): Promise<DatasetImportJob> {
  return apiFetch<DatasetImportJob>(
    `${scoped()}/datasets/imports/${encodeURIComponent(importId)}/cancel`,
    { method: 'POST' },
    signal,
  );
}

export function resumeDatasetImport(
  importId: string,
  signal?: AbortSignal,
): Promise<DatasetImportJob> {
  return apiFetch<DatasetImportJob>(
    `${scoped()}/datasets/imports/${encodeURIComponent(importId)}/resume`,
    { method: 'POST' },
    signal,
  );
}

/** Dry run → `DatasetUndoReport`; apply → 202 `DatasetImportJob`
 *  (`status: undoing`). */
export function undoDatasetImport(
  importId: string,
  body: DatasetUndoRequest,
  signal?: AbortSignal,
): Promise<DatasetUndoReport | DatasetImportJob> {
  return apiFetch<DatasetUndoReport | DatasetImportJob>(
    `${scoped()}/datasets/imports/${encodeURIComponent(importId)}/undo`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** A finished import's served `next_steps` entry, run as served: its
 *  `method` against its `path` under the project's prefix, no body
 *  (plan §8 question 10). */
export function runServedNextStep(
  step: NextStep,
  signal?: AbortSignal,
): Promise<unknown> {
  return apiFetch<unknown>(
    `${scoped()}${step.path}`,
    { method: step.method.toUpperCase() },
    signal,
  );
}

/** Batch Reprocess. `dry_run` defaults to true on the server; the caller
 *  always sends it explicitly. */
export function reprocessBatch(
  body: ReprocessRequest,
  signal?: AbortSignal,
): Promise<ReprocessResponse> {
  return apiFetch<ReprocessResponse>(
    `${scoped()}/reprocess`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

/** Single-item Reprocess; returns the post-write `items` (mapped like
 *  every other crop) for the caller to adopt. */
export async function reprocessCrop(
  cropId: string,
  body: ReprocessOneRequest,
  signal?: AbortSignal,
): Promise<ReprocessResponse<Crop>> {
  const res = await apiFetch<ReprocessResponse<RawCrop>>(
    `${scoped()}/crops/${encodeURIComponent(cropId)}/reprocess`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
  return { ...res, items: (res.items ?? []).map(mapRawCrop) };
}

/** Single-image Reprocess (`POST /images/{image_id}/reprocess`); returns
 *  the post-write `items` of that image, mapped like every other crop. */
export async function reprocessImage(
  imageId: string,
  body: ReprocessOneRequest,
  signal?: AbortSignal,
): Promise<ReprocessResponse<Crop>> {
  const res = await apiFetch<ReprocessResponse<RawCrop>>(
    `${scoped()}/images/${encodeURIComponent(imageId)}/reprocess`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
  return { ...res, items: (res.items ?? []).map(mapRawCrop) };
}

export function getReprocessJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<ReprocessJob> {
  return apiFetch<ReprocessJob>(
    `${scoped()}/reprocess/jobs/${encodeURIComponent(jobId)}`,
    {},
    signal,
  );
}

export function cancelReprocessJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<ReprocessJob> {
  return apiFetch<ReprocessJob>(
    `${scoped()}/reprocess/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: 'POST' },
    signal,
  );
}

/**
 * The structured W10 refusal (`{detail: ConfigErrorDetail}` with the
 * optional `issues`/`unmapped`/`import_id`), or `null` when the error
 * isn't one. The UI shows `message` and branches only on `error`.
 */
export function datasetErrorDetail(e: unknown): DatasetErrorDetail | null {
  if (!(e instanceof ApiError)) return null;
  const body = e.body;
  if (!body || typeof body !== 'object') return null;
  const detail = (body as { detail?: unknown }).detail;
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return null;
  const d = detail as Record<string, unknown>;
  if (typeof d.error !== 'string' || typeof d.message !== 'string') return null;
  return d as unknown as DatasetErrorDetail;
}

/** The served `message` of a W10 refusal, else the generic detail. */
export function datasetErrorText(e: unknown): string {
  const d = datasetErrorDetail(e);
  if (d) return d.message;
  if (e instanceof ApiError && e.detail) return e.detail;
  return (e as Error)?.message ?? String(e);
}

// -- Prompt packs (OpenProcessor W3; test-on-crop W5) --------------------
// any_domain_plan.md §3, §5.1, §7.2, §7.5;
// docs/design/w3-pack-editor-ui-plan-2026-09-27.md §1.

/** `GET /prompt_packs`: every pack (builtin, file, stored) plus the
 *  clone-only templates and the active ref. Also the W3 gate's probe. */
export function listPromptPacks(signal?: AbortSignal): Promise<PromptPackList> {
  return apiFetch<PromptPackList>(`${scoped()}/prompt_packs`, {}, signal);
}

export function getPromptPackSchema(signal?: AbortSignal): Promise<PromptPackSchema> {
  return apiFetch<PromptPackSchema>(`${scoped()}/prompt_packs/schema`, {}, signal);
}

/** `POST /prompt_packs/validate`: a draft's report. Never writes and
 *  never 422s; a reserved or taken `name` is an issue in the report. */
export function validatePromptPack(
  body: PackValidateRequest,
  signal?: AbortSignal,
): Promise<ValidationReport> {
  return apiFetch<ValidationReport>(
    `${scoped()}/prompt_packs/validate`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

export function getPromptPack(
  name: string,
  signal?: AbortSignal,
): Promise<PromptPackDoc> {
  return apiFetch<PromptPackDoc>(
    `${scoped()}/prompt_packs/${encodeURIComponent(name)}`,
    {},
    signal,
  );
}

export function getPromptPackRevisions(
  name: string,
  signal?: AbortSignal,
): Promise<ConfigRevisionList> {
  return apiFetch<ConfigRevisionList>(
    `${scoped()}/prompt_packs/${encodeURIComponent(name)}/revisions`,
    {},
    signal,
  );
}

export function getPromptPackRevision(
  name: string,
  revision: number,
  signal?: AbortSignal,
): Promise<PromptPackDoc> {
  return apiFetch<PromptPackDoc>(
    `${scoped()}/prompt_packs/${encodeURIComponent(name)}/revisions/${encodeURIComponent(String(revision))}`,
    {},
    signal,
  );
}

/** `POST /prompt_packs/{name}/clone` → 201 the new stored pack. */
export function clonePromptPack(
  name: string,
  body: ConfigCloneRequest,
): Promise<PromptPackDoc> {
  return apiFetch<PromptPackDoc>(
    `${scoped()}/prompt_packs/${encodeURIComponent(name)}/clone`,
    {
      method: 'POST',
      body: JSON.stringify(body),
    },
  );
}

/** `PUT /prompt_packs/{name}`: saves a new revision (OCC on
 *  `expected_revision`; 409 `revision_conflict` carries the current one). */
export function updatePromptPack(
  name: string,
  body: PackUpdateRequest,
): Promise<PromptPackDoc> {
  return apiFetch<PromptPackDoc>(`${scoped()}/prompt_packs/${encodeURIComponent(name)}`, {
    method: 'PUT',
    body: JSON.stringify(body),
  });
}

/** `DELETE /prompt_packs/{name}?expected_revision=` → 204. */
export function deletePromptPack(name: string, expectedRevision: number): Promise<void> {
  return apiFetch<void>(
    `${scoped()}/prompt_packs/${encodeURIComponent(name)}${qs({ expected_revision: expectedRevision })}`,
    {
      method: 'DELETE',
    },
  );
}

export function getActivePromptPack(signal?: AbortSignal): Promise<ActiveConfigResponse> {
  return apiFetch<ActiveConfigResponse>(`${scoped()}/prompt_packs/active`, {}, signal);
}

/** `POST /prompt_packs/{name}/activate` (OCC on `expected_active`). */
export function activatePromptPack(
  name: string,
  body: ConfigActivateRequest,
): Promise<ActivateResponse> {
  return apiFetch<ActivateResponse>(
    `${scoped()}/prompt_packs/${encodeURIComponent(name)}/activate`,
    {
      method: 'POST',
      body: JSON.stringify(body),
    },
  );
}

/** `POST /prompt_packs/active/rollback`: re-activates the previous pack. */
export function rollbackPromptPack(body: {
  expected_active: ActiveRef;
}): Promise<ActiveConfigResponse> {
  return apiFetch<ActiveConfigResponse>(`${scoped()}/prompt_packs/active/rollback`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// -- Region profiles and the config vocabulary (OpenProcessor W4) ---------
// any_domain_plan.md §4, §7.3, §7.4;
// docs/design/w4-profile-editor-ui-plan-2026-09-27.md §2.

/** `GET /region_profiles?include_templates=true`: every profile (env,
 *  registered, stored) plus the clone-only templates and the active ref.
 *  Also the W4 gate's probe. */
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
export function configErrorDetail(e: unknown): ConfigErrorDetail | null {
  if (!(e instanceof ApiError)) return null;
  const body = e.body;
  if (!body || typeof body !== 'object') return null;
  const detail = (body as { detail?: unknown }).detail;
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return null;
  const d = detail as Record<string, unknown>;
  if (typeof d.error !== 'string' || typeof d.message !== 'string') return null;
  return d as unknown as ConfigErrorDetail;
}

/** The served `message` of a config-store refusal, else the generic detail. */
export function configErrorText(e: unknown): string {
  const d = configErrorDetail(e);
  if (d) return d.message;
  if (e instanceof ApiError && e.detail) return e.detail;
  return (e as Error)?.message ?? String(e);
}
