/**
 * A small, JS-syntax-aware (not full-parser) scanner that pulls every
 * backend call the frontend makes directly out of the source, for
 * `endpointCatalog.test.ts` to check against the vendored OpenAPI
 * contract. Deliberately mechanical rather than hand-maintained: a new
 * `${scoped()}/...` call site is picked up automatically, and the
 * test's completeness guard fails loudly when this scanner can't make
 * sense of one instead of silently skipping it.
 *
 * Scope: understands template-literal URL composition
 * (`` `${scoped()}/foo/${id}` ``), the `qs({...})` query-string
 * helper (literal keys and `...spread` of a typed parameter), the
 * `.searchParams.set('key', ...)` pattern (`sse.ts`), and an
 * `options: { method: 'POST' }` object passed as the call's second
 * argument. It does NOT understand arbitrary imperative query-building
 * (e.g. `obj.key = value` assignment chains) — callers needing that
 * (there is exactly one, `getCluster`'s `cropQuery`) fall back to a
 * `Record<string, unknown>`-typed identifier, which resolves to
 * "params unchecked" rather than a wrong answer.
 */

export interface ApiCallSite {
  /** Raw path fragment as written, with dynamic segments replaced by `*`
   *  and the query string stripped — e.g. `/crops/*\/label`. */
  pathTemplate: string;
  /** HTTP method, defaulting to GET when no `method:` is found nearby
   *  (matches `apiFetch`'s own default). */
  method: string;
  /** Query parameter names sent, deduped. `null` when at least one
   *  query source could not be resolved (an untyped passthrough) —
   *  callers should skip the param check for that site rather than
   *  report a false positive. */
  queryParams: string[] | null;
  /** Byte offset of the template literal in the source, for error messages. */
  index: number;
  /** The literal source text matched, for error messages. */
  raw: string;
}

function skipString(src: string, i: number): number {
  const quote = src[i];
  i++;
  while (i < src.length) {
    if (src[i] === '\\') {
      i += 2;
      continue;
    }
    if (src[i] === quote) return i + 1;
    i++;
  }
  return i;
}

/** `src[i]` must be the opening character. Returns the index just past
 *  the matching closing character, skipping over nested strings/templates. */
function skipBalanced(src: string, i: number, open: string, close: string): number {
  let depth = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === '"' || c === "'") {
      i = skipString(src, i);
      continue;
    }
    if (c === '`') {
      i = skipTemplateLiteral(src, i);
      continue;
    }
    if (c === open) depth++;
    else if (c === close) {
      depth--;
      if (depth === 0) return i + 1;
    }
    i++;
  }
  return i;
}

function skipTemplateLiteral(src: string, i: number): number {
  i++; // past opening `
  while (i < src.length) {
    if (src[i] === '\\') {
      i += 2;
      continue;
    }
    if (src[i] === '`') return i + 1;
    if (src[i] === '$' && src[i + 1] === '{') {
      i = skipBalanced(src, i + 1, '{', '}');
      continue;
    }
    i++;
  }
  return i;
}

/**
 * Same length as `src`, with every comment, string literal and template
 * literal blanked out (replaced with spaces). Regexes that hunt for
 * *structural* code (`function foo(`, `interface Bar {`) run against
 * this instead of the raw source, so a JSDoc comment that happens to
 * mention "function" or a type name in prose can't be mistaken for a
 * real declaration — offsets stay valid for slicing back into `src`.
 */
function maskNonCode(src: string): string {
  const out = src.split('');
  const blank = (from: number, to: number) => {
    for (let k = from; k < to; k++) if (out[k] !== '\n') out[k] = ' ';
  };
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === '/' && src[i + 1] === '/') {
      const nl = src.indexOf('\n', i);
      const end = nl === -1 ? src.length : nl;
      blank(i, end);
      i = end;
      continue;
    }
    if (c === '/' && src[i + 1] === '*') {
      const close = src.indexOf('*/', i);
      const end = close === -1 ? src.length : close + 2;
      blank(i, end);
      i = end;
      continue;
    }
    if (c === '"' || c === "'") {
      const end = skipString(src, i);
      blank(i, end);
      i = end;
      continue;
    }
    if (c === '`') {
      const end = skipTemplateLiteral(src, i);
      blank(i, end);
      i = end;
      continue;
    }
    i++;
  }
  return out.join('');
}

/** Every backtick template literal in `src` whose text contains the
 *  literal `marker` (`${scoped()}` by default, or `${globalApi()}` for
 *  the small set of routes that stay global — P1 projects cutover),
 *  skipping comments and ordinary string literals so JSDoc examples
 *  don't count. */
export function findApiPrefixTemplates(
  src: string,
  marker: string = '${scoped()}',
): Array<{ start: number; end: number; text: string }> {
  const results: Array<{ start: number; end: number; text: string }> = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === '/' && src[i + 1] === '/') {
      const nl = src.indexOf('\n', i);
      i = nl === -1 ? src.length : nl;
      continue;
    }
    if (c === '/' && src[i + 1] === '*') {
      const end = src.indexOf('*/', i);
      i = end === -1 ? src.length : end + 2;
      continue;
    }
    if (c === '"' || c === "'") {
      i = skipString(src, i);
      continue;
    }
    if (c === '`') {
      const start = i;
      const end = skipTemplateLiteral(src, i);
      const text = src.slice(start, end);
      if (text.includes(marker)) {
        results.push({ start, end, text });
      }
      i = end;
      continue;
    }
    i++;
  }
  return results;
}

function splitTopLevel(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let cur = '';
  let i = 0;
  while (i < text.length) {
    const c = text[i];
    if (c === '"' || c === "'") {
      const j = skipString(text, i);
      cur += text.slice(i, j);
      i = j;
      continue;
    }
    if (c === '`') {
      const j = skipTemplateLiteral(text, i);
      cur += text.slice(i, j);
      i = j;
      continue;
    }
    if ('{[('.includes(c)) depth++;
    if ('}])'.includes(c)) depth--;
    if (c === ',' && depth === 0) {
      parts.push(cur);
      cur = '';
      i++;
      continue;
    }
    cur += c;
    i++;
  }
  if (cur.trim()) parts.push(cur);
  return parts;
}

function stripComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/(?<!:)\/\/[^\n]*/g, ' ');
}

/** Top-level `key: type` / `key?: type` member names of an
 *  interface/type-literal body (the text strictly between its `{` `}`). */
function extractInterfaceKeys(body: string): string[] {
  const clean = stripComments(body);
  const keys: string[] = [];
  for (const m of clean.matchAll(/(?:^|;|\n)\s*(\w+)\??\s*:/g)) {
    keys.push(m[1]);
  }
  return keys;
}

/** Resolves `identifier`'s declared type from the nearest `identifier:
 *  TypeName` occurrence before `beforeIndex`, then returns that named
 *  interface/type's member keys. `null` when the type can't be resolved
 *  to a closed member list (e.g. `Record<string, unknown>`, `unknown`) —
 *  an untyped passthrough the scanner correctly declines to guess at. */
function resolveIdentifierKeys(
  fullSrc: string,
  identName: string,
  beforeIndex: number,
): string[] | null {
  // Every structural regex below runs against `masked` (comments/strings/
  // templates blanked out, same length as `fullSrc`), so prose in a
  // JSDoc comment that happens to contain "function foo(" or a type name
  // can never be mistaken for a real declaration. Slicing for actual text
  // always uses `fullSrc` at the same offsets.
  const masked = maskNonCode(fullSrc);

  // Bound the search to the enclosing function's OWN parameter list —
  // not an arbitrary lookback window — so an identically-named parameter
  // on a different, earlier function in the same file can't be picked up
  // by accident.
  const fnRe = /\bfunction\s+\w+\s*\(/g;
  let fnMatch: RegExpExecArray | null = null;
  let m: RegExpExecArray | null;
  while ((m = fnRe.exec(masked))) {
    if (m.index >= beforeIndex) break;
    fnMatch = m;
  }
  if (!fnMatch) return null;
  const openParen = masked.indexOf('(', fnMatch.index);
  const closeParen = skipBalanced(masked, openParen, '(', ')');
  // Confirm `beforeIndex` actually falls inside THIS function's body —
  // otherwise `fnMatch` is some earlier, already-closed function and the
  // real enclosing function must be a form `fnRe` doesn't match (e.g. an
  // arrow function); fail closed rather than guess.
  let bodyStart = closeParen;
  while (bodyStart < masked.length && masked[bodyStart] !== '{') bodyStart++;
  if (bodyStart >= masked.length) return null;
  const bodyEnd = skipBalanced(masked, bodyStart, '{', '}');
  if (beforeIndex < closeParen || beforeIndex >= bodyEnd) return null;

  const windowStart = openParen;
  const maskedWindow = masked.slice(windowStart, closeParen);

  // Inline object-type annotation, e.g. `params: { page?: number; ... }`
  // — take the nearest one before `beforeIndex` and read its members
  // directly, same as a named interface.
  const inlineRe = new RegExp(`\\b${identName}\\s*:\\s*\\{`, 'g');
  const inlineMatches = [...maskedWindow.matchAll(inlineRe)];
  if (inlineMatches.length > 0) {
    const last = inlineMatches[inlineMatches.length - 1];
    const braceStart = windowStart + last.index! + last[0].length - 1;
    const braceEnd = skipBalanced(masked, braceStart, '{', '}');
    return extractInterfaceKeys(fullSrc.slice(braceStart + 1, braceEnd - 1));
  }

  const paramRe = new RegExp(
    `\\b${identName}\\s*:\\s*([A-Za-z_][A-Za-z0-9_<>[\\], .]*)`,
    'g',
  );
  const matches = [...maskedWindow.matchAll(paramRe)];
  if (matches.length === 0) return null;
  let typeName = matches[matches.length - 1][1].trim();
  typeName = typeName.split(/[<[]/)[0].trim();
  if (
    typeName === 'Record' ||
    typeName === 'unknown' ||
    typeName === '' ||
    typeName === 'object'
  ) {
    return null;
  }
  const blockRe = new RegExp(`(?:interface|type)\\s+${typeName}\\b[^{]*\\{`);
  const bm = blockRe.exec(masked);
  if (!bm) return null;
  const braceStart = masked.indexOf('{', bm.index);
  const braceEnd = skipBalanced(masked, braceStart, '{', '}');
  return extractInterfaceKeys(fullSrc.slice(braceStart + 1, braceEnd - 1));
}

/** Query keys sent by a `qs(<argText>)` call's argument text (without
 *  the outer parens). `null` if any source of keys is unresolvable. */
function extractQsKeys(
  fullSrc: string,
  argText: string,
  atIndex: number,
): string[] | null {
  const trimmed = argText.trim();
  const keys: string[] = [];
  if (trimmed.startsWith('{')) {
    const inner = trimmed.slice(1, trimmed.lastIndexOf('}'));
    for (const rawPart of splitTopLevel(inner)) {
      const part = rawPart.trim();
      if (!part) continue;
      if (part.startsWith('...')) {
        const ident = part.slice(3).trim();
        const resolved = resolveIdentifierKeys(fullSrc, ident, atIndex);
        if (resolved == null) return null;
        keys.push(...resolved);
        continue;
      }
      const m = /^([A-Za-z_$][\w$]*)/.exec(part);
      if (!m) return null;
      keys.push(m[1]);
    }
    return [...new Set(keys)];
  }
  // Bare identifier, possibly `as Record<string, unknown>`.
  const identMatch = /^([A-Za-z_$][\w$]*)/.exec(trimmed);
  if (!identMatch) return null;
  const resolved = resolveIdentifierKeys(fullSrc, identMatch[1], atIndex);
  return resolved == null ? null : [...new Set(resolved)];
}

/**
 * Parses every `${scoped()}`-containing template in `src` into a call
 * site: path template, method, and query params. `filePath` is only used
 * in error messages via `raw`.
 */
/** Simple `const NAME = '/literal';` string constants declared in `src`,
 *  so a path segment like `${REGION_BASE}` can be inlined to its actual
 *  value before path-template normalization, instead of collapsing to
 *  an opaque wildcard that loses its slash structure. */
function findStringConstants(src: string): Map<string, string> {
  const map = new Map<string, string>();
  for (const m of src.matchAll(/const\s+(\w+)(?::\s*string)?\s*=\s*'([^']*)';/g)) {
    map.set(m[1], m[2]);
  }
  return map;
}

export function scanApiCallSites(
  src: string,
  marker: string = '${scoped()}',
): ApiCallSite[] {
  const templates = findApiPrefixTemplates(src, marker);
  const consts = findStringConstants(src);
  const sites: ApiCallSite[] = [];

  for (const t of templates) {
    // `inner` keeps original offsets (needed to locate the `qs(...)` call
    // in `src`); `normInner` is a display copy with known local string
    // constants (e.g. `REGION_BASE`) inlined so a segment like
    // `${REGION_BASE}` normalizes to `/regions`, not an opaque `*`.
    const inner = t.text.slice(1, -1); // drop backticks
    let normInner = inner;
    for (const [name, value] of consts) {
      normInner = normInner.split(`\${${name}}`).join(value);
    }
    const markerIdx = inner.indexOf(marker);
    const afterPrefix = inner.slice(markerIdx + marker.length);
    let normAfterPrefix = normInner.slice(normInner.indexOf(marker) + marker.length);

    // Cut the path off at the query-string composition, if any.
    const qsMarker = '${qs(';
    const qsIdx = afterPrefix.indexOf(qsMarker);
    let queryParams: string[] | null = [];
    if (qsIdx !== -1) {
      const openParen =
        t.start + 1 + markerIdx + marker.length + qsIdx + qsMarker.length - 1;
      const closeParen = skipBalanced(src, openParen, '(', ')');
      const argText = src.slice(openParen + 1, closeParen - 1);
      queryParams = extractQsKeys(src, argText, t.start);
      const normQsIdx = normAfterPrefix.indexOf(qsMarker);
      normAfterPrefix = normAfterPrefix.slice(
        0,
        normQsIdx === -1 ? undefined : normQsIdx,
      );
    }

    // A literal query string composed by hand (no `qs()`), e.g.
    // `/thumbnail?size=${size}` on the getThumbUrl-style URL builders —
    // split it off and read its keys directly.
    const literalQIdx = normAfterPrefix.indexOf('?');
    if (queryParams != null && queryParams.length === 0 && literalQIdx !== -1) {
      const queryPart = normAfterPrefix.slice(literalQIdx + 1);
      const keys = [...queryPart.matchAll(/([A-Za-z_][A-Za-z0-9_]*)=/g)].map((m) => m[1]);
      if (keys.length > 0) queryParams = [...new Set(keys)];
      normAfterPrefix = normAfterPrefix.slice(0, literalQIdx);
    }

    const pathTemplate = normAfterPrefix.replace(/\$\{.*?\}/g, '*') || '/';

    // Method: look at the call's next argument (an options object), if any.
    let method = 'GET';
    let scanPos = t.end;
    while (scanPos < src.length && /\s/.test(src[scanPos])) scanPos++;
    if (src[scanPos] === ',') {
      scanPos++;
      while (scanPos < src.length && /\s/.test(src[scanPos])) scanPos++;
      if (src[scanPos] === '{') {
        const closeBrace = skipBalanced(src, scanPos, '{', '}');
        const optionsText = src.slice(scanPos, closeBrace);
        const m = /method\s*:\s*['"]([A-Za-z]+)['"]/.exec(optionsText);
        if (m) method = m[1].toUpperCase();
      }
    }

    // .searchParams.set('key', ...) pattern (sse.ts) — scan a bounded
    // window after the template for any such calls.
    const window = src.slice(t.end, Math.min(src.length, t.end + 1500));
    const setMatches = [...window.matchAll(/searchParams\.set\(\s*['"](\w+)['"]/g)];
    if (setMatches.length > 0) {
      queryParams = [...new Set(setMatches.map((m) => m[1]))];
    }

    sites.push({
      pathTemplate,
      method,
      queryParams,
      index: t.start,
      raw: t.text,
    });
  }

  return sites;
}
