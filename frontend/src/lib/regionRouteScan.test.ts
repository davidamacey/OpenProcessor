/**
 * CI ratchet: `api.ts`'s "plates browse / training-cohort selection"
 * sub-system composes its base path through the single `REGION_BASE`
 * constant, never a re-typed `/plates` (or `/regions`) literal.
 *
 * This is the frontend half of Wave 2's wire rename
 * (docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §8.4(i)):
 * OpenProcessor (openprocessor) renamed `/plates` -> `/regions` for this
 * whole route family (merged to `main` at `b3f928d`). `REGION_BASE` is
 * the one line that encodes the backend's current name for that base
 * path — flipping it is the entire lockstep change (C13 introduces the
 * constant with the value unchanged; C14 flips it). A future accidental
 * revert to a bare `/plates` literal anywhere in this family would
 * silently 404 against the live backend with no compile error and no
 * other failing test — this guard is what catches that.
 *
 * Sibling guards: `apiPrefixScan.test.ts` ratchets `${API_PREFIX}`
 * composition; `plateThumbUrlScan.test.ts` ratchets the
 * `region_thumbnail` path segment. This one ratchets the *base path*
 * of the `/plates`-family collection routes specifically.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const apiSrc = readFileSync(path.resolve(here, 'api.ts'), 'utf-8');

/** Local copy of apiPrefixScan.test.ts's helper — deliberately NOT
 *  imported from that sibling `.test.ts` file, which would re-register
 *  its whole describe/it tree as a side effect of the module import
 *  and double-run it under `vitest run`. */
function stripComments(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/(?<!:)\/\/[^\n]*/g, ' ');
}

describe('api.ts declares exactly one REGION_BASE constant', () => {
  it('exports/declares REGION_BASE with the current backend value', () => {
    const m = apiSrc.match(/const REGION_BASE = '([^']*)';/);
    expect(m).not.toBeNull();
    // Pinned to the backend's CURRENT name. When OpenProcessor renames
    // this route family, update the value here in the SAME commit that
    // flips the constant (C14) — this assertion is the deliberate,
    // single, expected diff line of that lockstep change.
    expect(m![1]).toBe('/regions');
  });
});

describe('no bare /plates route-family literal survives outside REGION_BASE', () => {
  /**
   * Matches a `/plates` (or a stray `/regions`, post-flip the ONLY
   * sanctioned spelling is via the constant) literal in URL-composition
   * position within the "plates browse" call sites — i.e. immediately
   * after `${API_PREFIX}` or `${REGION_BASE}` is exactly what's allowed;
   * a raw `${API_PREFIX}/plates...` re-introduction is what this catches.
   */
  const BARE_REGION_LITERAL = /\$\{API_PREFIX\}\/(?:plates|regions)(?:[/'"`]|\$)/;

  it('finds zero offenders in api.ts (excluding the REGION_BASE declaration itself)', () => {
    const lines = stripComments(apiSrc)
      .split('\n')
      .filter((line) => !/const REGION_BASE = /.test(line));
    const offenders = lines.filter((line) => BARE_REGION_LITERAL.test(line));
    expect(offenders).toEqual([]);
  });

  it('every plates-family call site in api.ts composes through ${API_PREFIX}${REGION_BASE}', () => {
    const CALLS = [
      /\$\{API_PREFIX\}\$\{REGION_BASE\}\$\{qs\(params as Record<string, unknown>\)\}`/,
      /\$\{API_PREFIX\}\$\{REGION_BASE\}\/cluster\$\{qs\(\{/,
      /\$\{API_PREFIX\}\$\{REGION_BASE\}\/cluster\/status`/,
      /\$\{API_PREFIX\}\$\{REGION_BASE\}\/clusters\/refine\/\$\{clusterId\}`/,
      /\$\{API_PREFIX\}\$\{REGION_BASE\}\/clusters\$\{qs\(\{/,
      /\$\{API_PREFIX\}\$\{REGION_BASE\}\/fp_centroids\/build`/,
      /\$\{API_PREFIX\}\$\{REGION_BASE\}\/fp_centroids\/status`/,
      /\$\{API_PREFIX\}\$\{REGION_BASE\}\/suspected_false_positives\$\{qs\(\{/,
      /\$\{API_PREFIX\}\$\{REGION_BASE\}\/training_candidates\$\{qs\(\{ mode, \.\.\.params \}\)\}`/,
      /\$\{API_PREFIX\}\$\{path\}`,\s*\n\s*\{\s*\n\s*method: 'POST',\s*\n\s*body: JSON\.stringify\(\{\s*\n\s*crop_ids/,
    ];
    for (const re of CALLS) {
      expect(apiSrc).toMatch(re);
    }
  });
});
