/**
 * Shared helpers for this repo's source-text-scan regression tests
 * (docs/design/test-audit-2026-09-24.md T1/P2-2). A scan matching exact
 * whitespace/indentation is a false-positive machine: it fails on a
 * harmless prettier re-wrap and silently passes on a real behavior
 * regression that happens to keep the same words in a different shape.
 * These helpers make a scan robust to reformatting without weakening
 * what it actually checks.
 *
 * `stripComments` moved here from `apiPrefixScan.test.ts` (its original
 * home) so every scan can share one implementation instead of each
 * hand-rolling (or copy-pasting, as `regionRouteScan.test.ts` used to)
 * its own.
 */

/** Strip block comments, HTML comments and line comments (but not a
 *  `://` inside a URL) before scanning source text. */
export function stripComments(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/(?<!:)\/\/[^\n]*/g, ' ');
}

/**
 * Comment-stripped, whitespace-collapsed source: every run of
 * whitespace (including newlines) becomes a single space. Use this
 * before a literal-substring or regex match that shouldn't care whether
 * the matched code is on one line, wrapped across several, or indented
 * with 2 vs. 4 spaces.
 */
export function normalize(src: string): string {
  return stripComments(src).replace(/\s+/g, ' ').trim();
}

/**
 * Brace-balanced function-body extraction, indentation-independent.
 *
 * Replaces the fragile `/function X\([\s\S]*?\n {2}\}/`-style regexes
 * that broke on a prettier re-wrap (test-audit-2026-09-24.md T1,
 * `clusterMoveRace.test.ts:267,279,312`): those assume the closing
 * brace sits at column 2 on its own line. This instead finds
 * `function <name>(` (optionally `async`), then walks forward counting
 * `{`/`}` depth from the function's own opening brace to find its
 * matching close, so the extraction survives any reformatting that
 * keeps the function's braces balanced.
 *
 * Returns the whole match (signature line through the closing `}`), or
 * `null` if the function isn't found.
 */
export function extractFunction(src: string, name: string): string | null {
  const sigPattern = new RegExp(
    `(?:async\\s+)?function\\s+${name}\\s*\\([^)]*\\)[^{]*\\{`,
  );
  return extractBalanced(src, sigPattern);
}

/**
 * Generalization of `extractFunction` for a construct that isn't a
 * `function` declaration — an arrow-function call argument
 * (`obj.method(async (...) => { ... })`), an object/array literal
 * passed as a call argument, etc. `startPattern` must match through
 * (and include) the opening `{` the brace-balance walk should start
 * counting from; everything from that match's start through the
 * matching close brace is returned.
 */
export function extractBalanced(src: string, startPattern: RegExp): string | null {
  const match = startPattern.exec(src);
  if (!match) return null;
  const start = match.index;
  const openIdx = start + match[0].length - 1;
  if (src[openIdx] !== '{') return null;
  let depth = 0;
  for (let i = openIdx; i < src.length; i++) {
    const ch = src[i];
    if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return src.slice(start, i + 1);
    }
  }
  return null;
}
