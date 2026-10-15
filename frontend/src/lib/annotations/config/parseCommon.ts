/**
 * Shared primitives for the tier-2 slot-config parser: the shape and
 * identifier checks every capability parser uses, the parse context
 * threaded through a document, and the `extras` passthrough.
 */

import { LIMITS, IDENTIFIER_RE, WIRE_FIELD_RE, FORBIDDEN_KEYS } from './allowLists';

export interface ParseContext {
  /** Combos already claimed by a previously-resolved slot, mapped to
   *  the action that claimed them. Cross-slot collisions on a DIFFERENT
   *  action reject; the same action is fine. */
  claimedCombos?: Map<string, string>;
  /** `?tab=` values already in use — core review tabs and preset ids as
   *  well as earlier slots. */
  takenUrlIds?: Set<string>;
  takenEndpointIds?: Set<string>;
  takenKeys?: Set<string>;
}

/* ------------------------------------------------------------------ */
/* Generic helpers                                                     */
/* ------------------------------------------------------------------ */

export function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

export function hasForbiddenKey(v: unknown): boolean {
  if (Array.isArray(v)) return v.some((x) => hasForbiddenKey(x));
  if (isPlainObject(v)) {
    for (const k of Object.keys(v)) {
      if (FORBIDDEN_KEYS.has(k)) return true;
      if (hasForbiddenKey(v[k])) return true;
    }
    return false;
  }
  return false;
}

export function isWireField(v: unknown): v is string {
  return (
    typeof v === 'string' &&
    v.length > 0 &&
    v.length <= LIMITS.identifierChars &&
    WIRE_FIELD_RE.test(v)
  );
}

export function isIdentifier(v: unknown): v is string {
  return (
    typeof v === 'string' &&
    v.length > 0 &&
    v.length <= LIMITS.identifierChars &&
    IDENTIFIER_RE.test(v)
  );
}

export function isLabel(v: unknown): v is string {
  return typeof v === 'string' && v.length > 0 && v.length <= LIMITS.labelChars;
}

/* ------------------------------------------------------------------ */
/* extras                                                              */
/* ------------------------------------------------------------------ */

function depthOf(v: unknown, depth = 0): number {
  if (depth > 8) return depth;
  if (Array.isArray(v)) {
    return v.length ? Math.max(...v.map((x) => depthOf(x, depth + 1))) : depth;
  }
  if (isPlainObject(v)) {
    const keys = Object.keys(v);
    return keys.length ? Math.max(...keys.map((k) => depthOf(v[k], depth + 1))) : depth;
  }
  return depth;
}

/** Builds a fresh object/array field-by-field — deliberately not a
 *  `JSON.parse(JSON.stringify())` shortcut, so the result can never
 *  carry an own `__proto__`/`constructor`/`prototype` key even if a
 *  future caller forgets the earlier `hasForbiddenKey` guard. */
function cloneExtras(v: unknown): unknown {
  if (Array.isArray(v)) return v.map((x) => cloneExtras(x));
  if (isPlainObject(v)) {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(v)) {
      if (FORBIDDEN_KEYS.has(k)) continue;
      out[k] = cloneExtras(v[k]);
    }
    return out;
  }
  return v;
}

export function parseExtras(
  raw: unknown,
  key: string,
  errors: string[],
): Record<string, unknown> | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw)) {
    errors.push(`slot "${key}": extras must be a plain object — skipped`);
    return null;
  }
  if (hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": extras contains a forbidden key — skipped`);
    return null;
  }
  if (depthOf(raw) > 8) {
    errors.push(`slot "${key}": extras exceeds the maximum nesting depth of 8 — skipped`);
    return null;
  }
  return cloneExtras(raw) as Record<string, unknown>;
}
