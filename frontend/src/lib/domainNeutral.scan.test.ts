/**
 * CI ratchet for the domain-neutral directive
 * (docs/design/domain-neutral-audit-2026-09-24.md §7.2, §8): Cropwright must
 * work for any data domain, so nothing under `src/` — code, UI strings,
 * comments — names one domain (license plates, vehicles). Comments count
 * too: the scan reads raw source text. (Deployment-specific names and
 * numbers are caught by a separate, repo-wide leak gate, not here.)
 *
 * `ALLOWED` is exact and per-file, each entry with the reason it is
 * allowed. Domain profiles live under `examples/` (outside `src/`, never
 * bundled), so most entries are this file, the one test that exercises
 * those examples, and test fixtures that mirror a real backend response
 * verbatim (a served class/model name is approved content — see each
 * entry's reason). Never add an entry to make a red run green — fix the
 * file.
 *
 * Mutation-check: add a `// plate` line to a scratch copy of any product
 * file under `src/` and this test goes red.
 *
 * Also scanned: the `LPR`/`lpr` abbreviation (audit §7.1 grep 1) and the
 * §8 car/vehicle sweep — our own hardcoded copy and test fixture class
 * names, not served data (see VEHICLE_DOMAIN_PATTERN below for exactly
 * which words and why). `Gemma`/`SAM3` are deliberately NOT scanned here:
 * per owner direction, a served model/detector id (`gemma-4-e4b`, `sam3`,
 * their vocabulary-served display labels) is approved content, and nearly
 * every remaining occurrence of those two words in `src/` is exactly that
 * — a test fixture mirroring `GET {API_PREFIX}/regions/vocabulary`'s
 * `chain_actors`. A bare-word scan for them would flag those fixtures,
 * not a real leak, and balloon this allow-list for no signal; our own
 * hardcoded copy naming them was fixed by hand instead (audit §8).
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const srcRoot = path.resolve(here, '..');

/** One domain's noun and its model abbreviation. The lookbehind skips
 *  `template`/`Template`. */
const DOMAIN_PATTERN = /(?<!tem)plate|\blpr\b|lpr_/i;
/** §8 sweep: the car/vehicle domain words our own copy and test fixtures
 *  used to hardcode (audit §8 grep: `vehicle`, `sedan|suv|bmw|motorcycle`).
 *  Word-bounded so it doesn't false-positive on unrelated words (e.g.
 *  `\bcar\b` doesn't match `cargo`/`cars`/`scar`); `classic_car`/
 *  `sports_car`/`dumptruck` are matched as substrings since they're
 *  compound identifiers, not standalone words. */
const VEHICLE_DOMAIN_PATTERN =
  /\b(vehicle|sedan|suvs?|motorcycle|bmw|audi|honda|porsche|subaru|pickup|coupe|sidecar|car)\b|classic_car|sports_car|dumptruck/i;

/** `src/`-relative path -> why it may still match. */
const ALLOWED: Record<string, string> = {
  'lib/domainNeutral.scan.test.ts': 'this file: it spells out the patterns it scans for',
  'lib/annotations/profiles.falsification.test.ts':
    'the one test that exercises the example profiles under examples/ (audit §3.9 KEEP-example)',
  'lib/test/fixtures/trainRun.ts':
    "real served fixtures captured verbatim from a live train run (see the file header) — the class names in its per-class table are that deployment's own registry, not domain fiction",
};

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

function rel(file: string): string {
  return path.relative(srcRoot, file).split(path.sep).join('/');
}

function offendingLines(file: string): string[] {
  return readFileSync(file, 'utf-8')
    .split('\n')
    .map((line, i) => ({ line, n: i + 1 }))
    .filter(({ line }) => DOMAIN_PATTERN.test(line) || VEHICLE_DOMAIN_PATTERN.test(line))
    .map(({ line, n }) => `${rel(file)}:${n}: ${line.trim()}`);
}

// Product source and tests alike: a test fixture in one domain is a
// domain leak too (audit §4.4 — tests use the neutral widget/tag domain).
const scanned = walk(srcRoot).filter((f) => /\.(ts|svelte|js|json|css|html)$/.test(f));

describe('domain-neutral source (audit §7.2)', () => {
  it('scans a real tree', () => {
    expect(scanned.length).toBeGreaterThan(100);
  });

  it('names no single data domain outside the allow-list', () => {
    const offenders = scanned
      .filter((f) => !(rel(f) in ALLOWED))
      .flatMap((f) => offendingLines(f));
    expect(offenders).toEqual([]);
  });

  it('every allow-listed file exists and still needs its entry', () => {
    for (const file of Object.keys(ALLOWED)) {
      const full = path.join(srcRoot, file);
      expect(
        offendingLines(full),
        `${file} no longer matches; drop it from ALLOWED`,
      ).not.toEqual([]);
    }
  });
});
