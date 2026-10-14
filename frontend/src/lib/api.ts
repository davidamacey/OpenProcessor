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

import {
  ApiError,
  RegionProfileUnavailableError,
  structuredErrorCode,
} from '$lib/api_errors';
import {
  isNoRegionProfileDetail,
  notifyRegionProfileUnavailable,
} from '$lib/regionProfileUnavailable';

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
function staleProjectError(): DOMException {
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
  // Reads: 1 initial + 3 retries on 5xx => 4 attempts max. A write may
  // already have been executed server-side (a 504 or a lost response), so it
  // is sent exactly once.
  const method = (init.method ?? 'GET').toUpperCase();
  const maxAttempts =
    method === 'GET' || method === 'HEAD' ? RETRY_DELAYS_MS.length + 1 : 1;
  for (; attempt < maxAttempts; attempt++) {
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
      if (res.status === 409 && isNoRegionProfileDetail(structuredErrorCode(body))) {
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
    if (attempt < maxAttempts - 1) {
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

export function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
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
