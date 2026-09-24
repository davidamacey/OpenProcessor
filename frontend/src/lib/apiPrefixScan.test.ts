/**
 * CI ratchet: every backend URL in this app composes through
 * `API_PREFIX` (`src/lib/api.ts`), never through a hardcoded `/curation` or
 * `/curation` literal.
 *
 * This is the frontend half of H1 (coordination plan §5.3 — see
 * `docs/design/backend-integration-phase-d-static-plan-2026-09-20.md`).
 * `T-B4` made the existing tests prefix-portable, which is a different
 * property: a suite that derives its expectations from `API_PREFIX`
 * still passes if a new call site hardcodes the prefix. These two
 * guards fail instead.
 *
 * Guard 1 (`no bare prefix literal`) is name-specific: it bans the two
 * literals we know about. Guard 2 (`composition ratchet`) is
 * name-agnostic: it requires the composition *shape* regardless of
 * which string someone reaches for. Keep both — neither subsumes the
 * other.
 *
 * Sibling guard: `plateThumbUrlScan.test.ts` pins the region-thumbnail
 * path *segment*. Different concern, different exclusions.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const libRoot = here;
const srcRoot = path.resolve(here, '..');

/** The one file allowed to name a prefix literal — see guard 2b. */
const PREFIX_DECLARATION_FILE = path.join('lib', 'api.ts');

/**
 * Matches a prefix literal in URL-composition position: the prefix
 * opens a string literal, or directly follows a closed `${…}`
 * interpolation.
 *
 * Anchoring on the quote/brace is what separates a URL from prose. The
 * repo has five executable strings that *mention* an endpoint path
 * ("failed to load /curation/methods", a `title=` tooltip, …); in all of them
 * the prefix is preceded by a space or `>`, so none matches. A reworded
 * message that opens with the path (`'/curation/methods is down'`) WOULD
 * trip this — reword it, don't loosen the pattern.
 */
const BARE_PREFIX_PATTERN = /(?:['"`]|\})\/(?:op|curation)(?:[/'"`]|\$)/;

/**
 * Strip comments before scanning. Without this the guard flags 12 files
 * on a clean tree: the repo documents backend routes in JSDoc using
 * backtick-quoted paths (`` `/curation/methods` ``), which is indistinguishable
 * from a code literal by regex alone. Those comments are accurate prose
 * and are NOT in scope to rewrite.
 *
 * The `(?<!:)` guard keeps `http://…` intact — without it the line
 * comment rule truncates `` `http://${host}:${s.port}` ``
 * (`components/MonitoringLinks.svelte:33`). Over-stripping can only
 * cause a missed violation, never a false alarm, but there is no reason
 * to accept even that.
 */
export function stripComments(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/(?<!:)\/\/[^\n]*/g, ' ');
}

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

const scanned = walk(srcRoot)
  .filter((f) => (f.endsWith('.ts') || f.endsWith('.svelte')) && !f.endsWith('.test.ts'))
  .map((f) => path.relative(srcRoot, f))
  .sort();

describe('stripComments (the guard is only as good as this)', () => {
  it('strips JSDoc blocks', () => {
    expect(stripComments('/** hits `/curation/methods` */\nconst a = 1;')).not.toMatch(
      BARE_PREFIX_PATTERN,
    );
  });

  it('strips line comments', () => {
    expect(stripComments("// see '/curation/crops'\nconst a = 1;")).not.toMatch(
      BARE_PREFIX_PATTERN,
    );
  });

  it('strips Svelte markup comments', () => {
    expect(stripComments("<!-- '/curation/clusters' -->\n<div />")).not.toMatch(
      BARE_PREFIX_PATTERN,
    );
  });

  it('does not truncate a protocol-relative URL at its double slash', () => {
    expect(stripComments('const u = `http://${host}:${port}/x`;')).toContain('${port}/x');
  });

  it('leaves a real code literal alone', () => {
    expect(stripComments("apiFetch('/curation/classes');")).toMatch(BARE_PREFIX_PATTERN);
  });
});

describe('no bare /curation or /curation literal composes a URL outside api.ts', () => {
  it('scans the whole non-test source tree (sanity check the walk)', () => {
    expect(scanned.length).toBeGreaterThan(80);
    expect(scanned).toContain(path.join('lib', 'api.ts'));
    expect(scanned).toContain(path.join('lib', 'sse.ts'));
    expect(scanned).toContain(path.join('routes', 'export', '+page.svelte'));
    expect(scanned).toContain(
      path.join('lib', 'annotations', 'profiles', 'licensePlate.ts'),
    );
  });

  it('finds zero offenders', () => {
    const offenders: string[] = [];
    for (const rel of scanned) {
      if (rel === PREFIX_DECLARATION_FILE) continue;
      const lines = stripComments(readFileSync(path.join(srcRoot, rel), 'utf-8')).split(
        '\n',
      );
      lines.forEach((line, i) => {
        if (BARE_PREFIX_PATTERN.test(line))
          offenders.push(`${rel}:${i + 1}  ${line.trim()}`);
      });
    }
    expect(offenders).toEqual([]);
  });
});

const apiSrc = readFileSync(path.resolve(libRoot, 'api.ts'), 'utf-8');
const sseSrc = readFileSync(path.resolve(libRoot, 'sse.ts'), 'utf-8');

describe('api.ts composition ratchet — every apiFetch path starts with ${API_PREFIX}', () => {
  /**
   * Call sites are written across several lines and sometimes carry an
   * interleaved `//` comment before the path argument (e.g. api.ts's
   * getCluster). Capture the first 28 characters after the opening
   * paren, skipping whitespace and line comments, and require the path
   * to open with the interpolation.
   */
  const CALL = /apiFetch(?:<[^>]*>)?\(\s*(?:\/\/[^\n]*\n\s*)*(.{0,28})/g;

  const heads = [...apiSrc.matchAll(CALL)].map((m) => m[1]);
  // Two of these are the declaration itself (`apiFetch<T>(\n  path: string`)
  // and its internal recursion guard, whose first argument is named `path`.
  const declarations = heads.filter((h) => h.startsWith('path'));
  const calls = heads.filter((h) => !h.startsWith('path'));

  it('finds the expected population (fails loudly if the regex stops matching)', () => {
    expect(declarations.length).toBe(2);
    expect(calls.length).toBeGreaterThanOrEqual(70);
  });

  it('every call composes its path from API_PREFIX', () => {
    const bad = calls.filter((h) => !h.startsWith('`${API_PREFIX}'));
    expect(bad).toEqual([]);
  });
});

describe('${apiBase} is always followed by ${API_PREFIX}', () => {
  /**
   * The eight `${apiBase}…` template builders in api.ts bypass
   * `apiFetch` entirely (thumbnail/image/registry URL helpers), as do
   * sse.ts's four EventSource URLs and export/+page.svelte's raw
   * `fetch`. They are exactly the sites the original P4.1 spec
   * undercounted, so pin them by shape.
   *
   * `resolveApiUrl` is the one legitimate exception: it prepends
   * `apiBase` to a path the SERVER already emitted complete with its
   * own prefix (see api.ts's doc comment above it).
   */
  const files: [string, string][] = [
    ['lib/api.ts', apiSrc],
    ['lib/sse.ts', sseSrc],
    [
      'routes/export/+page.svelte',
      readFileSync(path.resolve(srcRoot, 'routes/export/+page.svelte'), 'utf-8'),
    ],
  ];

  it('has exactly one exception, and it is resolveApiUrl', () => {
    const bad: string[] = [];
    for (const [rel, src] of files) {
      const stripped = stripComments(src);
      for (const m of stripped.matchAll(/\$\{apiBase\}(.{0,16})/g)) {
        if (!m[1].startsWith('${API_PREFIX}')) bad.push(`${rel}  \${apiBase}${m[1]}`);
      }
    }
    expect(bad).toEqual(['lib/api.ts  ${apiBase}${url}`;']);
  });

  it('sse.ts builds both EventSource URLs from API_PREFIX', () => {
    expect([
      ...sseSrc.matchAll(/\$\{apiBase\}\$\{API_PREFIX\}\/(pipeline\/events|events)/g),
    ]).toHaveLength(4);
  });
});

describe('api.ts confines its prefix literal to normalizeApiPrefix', () => {
  const lines = stripComments(apiSrc).split('\n');
  const hits = lines
    .map((line, i) => [i + 1, line.trim()] as const)
    .filter(([, line]) => BARE_PREFIX_PATTERN.test(line));

  it('has exactly one executable prefix literal', () => {
    expect(hits).toHaveLength(1);
  });

  // Flipped from '/curation' at T-E2. Every other assertion in this file is
  // prefix-name-agnostic.
  it('and it is normalizeApiPrefix’s /curation fallback', () => {
    expect(hits[0]![1]).toMatch(
      /^if \(!trimmed \|\| trimmed\.startsWith\('__'\)\) return '\/curation';$/,
    );
  });

  it('exports normalizeApiPrefix so the flip has exactly one edit point', () => {
    expect(apiSrc).toMatch(/export function normalizeApiPrefix\(/);
  });
});
