/**
 * Compiles a `{API_PREFIX}`-relative path template into the closure the
 * `SlotSpec` type expects (contract §5.1/§5.2).
 *
 * The ONLY transformation is `String.replaceAll` of an allow-listed
 * placeholder with `encodeURIComponent(value)`. There is no expression
 * evaluation, no dynamic `Function` construction, no interpolation of anything that did
 * not come from the caller's own argument list. A placeholder outside
 * the allow-list rejects the template rather than being interpreted —
 * `{cropId.toUpperCase()}` is not a clever path, it is an attack, and
 * the parser must say so.
 */

import { LIMITS, PATH_PLACEHOLDERS } from './allowLists';

/** Anything that can smuggle a different origin, escape the path, or
 *  break out of an attribute. Checked before placeholder extraction so
 *  a hostile placeholder name cannot hide inside one. */
const UNSAFE_PATH_FRAGMENT = /(:\/\/|\.\.|\\|\s|[<>"'`#%])/;

/** Path minus its placeholders must look like a path. */
const PATH_SHAPE = /^\/[A-Za-z0-9._~\-/]*(\?[A-Za-z0-9._~\-=&]*)?$/;

export interface PathTemplateResult {
  /** Present only when the template is legal. */
  template?: string;
  /** Placeholder names actually used, in first-appearance order. */
  placeholders?: string[];
  error?: string;
}

export function validatePathTemplate(
  raw: unknown,
  allowed: readonly string[] = PATH_PLACEHOLDERS,
): PathTemplateResult {
  if (typeof raw !== 'string' || raw.length === 0) {
    return { error: 'path must be a non-empty string' };
  }
  if (raw.length > LIMITS.pathChars) {
    return { error: `path exceeds ${LIMITS.pathChars} characters` };
  }
  if (!raw.startsWith('/') || raw.startsWith('//')) {
    return {
      error: `path must be API_PREFIX-relative and start with a single "/": ${raw}`,
    };
  }
  if (UNSAFE_PATH_FRAGMENT.test(raw)) {
    return { error: `path contains an unsafe fragment: ${raw}` };
  }
  const placeholders: string[] = [];
  const stripped = raw.replace(/\{([^{}]*)\}/g, (_m, name: string) => {
    if (!allowed.includes(name)) {
      placeholders.push(`\u0000${name}`); // marker: rejected below
    } else if (!placeholders.includes(name)) {
      placeholders.push(name);
    }
    return '_';
  });
  const bad = placeholders.find((p) => p.startsWith('\u0000'));
  if (bad) {
    return {
      error: `path placeholder "{${bad.slice(1)}}" is not in the allow-list [${allowed.join(', ')}]`,
    };
  }
  if (stripped.includes('{') || stripped.includes('}')) {
    return { error: `path has unbalanced braces: ${raw}` };
  }
  if (!PATH_SHAPE.test(stripped)) {
    return { error: `path has an illegal shape: ${raw}` };
  }
  return { template: raw, placeholders };
}

/** Substitutes allow-listed placeholders. Values are always
 *  `encodeURIComponent`'d, matching what every hand-written profile's
 *  path functions do. */
export function renderPathTemplate(
  template: string,
  values: Record<string, string | number>,
): string {
  let out = template;
  for (const [name, value] of Object.entries(values)) {
    out = out.replaceAll(`{${name}}`, encodeURIComponent(String(value)));
  }
  return out;
}
