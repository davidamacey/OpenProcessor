/**
 * Error shapes of the curation API: `ApiError`, the served-detail text
 * (`apiErrorText`) and the structured-detail reader. A leaf: imports no
 * other `api*` module, so the HTTP core can depend on it.
 */
import { REGION_PROFILE_UNAVAILABLE_MESSAGE } from '$lib/regionProfileUnavailable';

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

/** The `detail.error` code of a structured `{detail: {error, ...}}` body. */
export function structuredErrorCode(body: unknown): string | null {
  const d = (body as { detail?: unknown } | null)?.detail;
  if (!d || typeof d !== 'object' || Array.isArray(d)) return null;
  const code = (d as Record<string, unknown>).error;
  return typeof code === 'string' ? code : null;
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
    // Structured FastAPI details (`{detail: {error, message, ...}}`): the
    // served `message` is the sentence to show; `error` is the machine code
    // and only stands in when no message was served.
    if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
      const rec2 = raw as Record<string, unknown>;
      raw =
        typeof rec2.message === 'string' && rec2.message
          ? rec2.message
          : (rec2.error ?? null);
    }
  }
  if (typeof raw !== 'string') return null;
  const text = raw.trim();
  if (!text) return null;
  return text.length > DETAIL_MAX_CHARS
    ? `${text.slice(0, DETAIL_MAX_CHARS - 1)}…`
    : text;
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

/**
 * A structured `{detail: {error, message, ...}}` refusal, or `null` when the
 * error is not one (a network failure, a plain-string detail, a pydantic
 * validation list). The UI renders `message` verbatim and branches only on
 * the served `error` code.
 */
export function structuredDetail<T extends { error: string; message: string }>(
  e: unknown,
): T | null {
  if (!(e instanceof ApiError)) return null;
  const detail = (e.body as { detail?: unknown } | null)?.detail;
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return null;
  const d = detail as Record<string, unknown>;
  if (typeof d.error !== 'string' || typeof d.message !== 'string') return null;
  return d as unknown as T;
}

/** The text to show for a failed call: the served structured `message`, else
 *  the generic `ApiError.detail` (a plain string or a joined validation
 *  list), else the error's own message. Never the machine code of a
 *  structured detail that carries a message. */
export function apiErrorText(e: unknown): string {
  const d = structuredDetail<{ error: string; message: string }>(e);
  if (d) return d.message;
  if (e instanceof ApiError && e.detail) return e.detail;
  return (e as Error)?.message ?? String(e);
}
