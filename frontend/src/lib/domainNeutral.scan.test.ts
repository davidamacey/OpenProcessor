/**
 * CI ratchet for the domain-neutral directive
 * (docs/design/domain-neutral-audit-2026-09-24.md §7.2): Cropwright must
 * work for any data domain, so nothing under `src/` — code, UI strings,
 * comments — names one domain (license plates) or the private deployment
 * it came from. Comments count too: the scan reads raw source text.
 *
 * `ALLOWED` is exact and per-file, each entry with the reason it is still
 * allowed. The example profiles move to `examples/` (audit steps 7-11,
 * which wait on the backend's naming-w2); every entry below goes away with
 * the step named in its reason. Never add an entry to make a red run
 * green — fix the file.
 *
 * Mutation-check: add a `// plate` line to a scratch copy of any product
 * file under `src/` and this test goes red.
 *
 * Not yet scanned: the `LPR`/`lpr` abbreviation, which the audit's §8
 * sweep covers separately.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const srcRoot = path.resolve(here, '..');

/** One domain's noun. The lookbehind skips `template`/`Template`. */
const DOMAIN_PATTERN = /(?<!tem)plate/i;
/** The private origin: its name, and its live dataset numbers/crop ids. */
const PRIVATE_PATTERN = /legacy|1,000|00000000/i;
/** The private stack's retired `/curation` prefix and `op_` names. Case-
 *  sensitive: an uppercase `KB` is a size unit. */
const PRIVATE_OP_PATTERN = /\bkb\b/;

/** `src/`-relative path -> why it may still match. */
const ALLOWED: Record<string, string> = {
  'lib/domainNeutral.scan.test.ts': 'this file: it spells out the patterns it scans for',
  'lib/annotations/profiles/licensePlate.ts':
    'the license-plate example profile; moves to examples/ in audit step 9',
  'lib/annotations/profiles/aircraftTailNumber.ts':
    'a demo profile that describes itself against the license-plate example; moves to examples/ in audit step 9',
  'lib/annotations/profiles/defectCode.ts':
    'a demo profile that describes itself against the license-plate example; moves to examples/ in audit step 9',
  'lib/annotations/registeredSlots.ts':
    'builtinSlots still registers the license-plate example profile until audit step 9 empties it',
  'lib/test/fixtures/regionSlot.ts':
    "reuses the example profile's region_* wire map until audit step 8's regionSlotFromServedProfile",
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
    .filter(
      ({ line }) =>
        DOMAIN_PATTERN.test(line) ||
        PRIVATE_PATTERN.test(line) ||
        PRIVATE_OP_PATTERN.test(line),
    )
    .map(({ line, n }) => `${rel(file)}:${n}: ${line.trim()}`);
}

const files = walk(srcRoot).filter((f) => /\.(ts|svelte|js|json|css|html)$/.test(f));
// Step 5 of the audit covers product source; the test files follow in
// step 6.
const scanned = files.filter((f) => !/\.test\.ts$/.test(f));

describe('domain-neutral source (audit §7.2)', () => {
  it('scans a real tree', () => {
    expect(scanned.length).toBeGreaterThan(100);
  });

  it('names no single data domain and no private origin outside the allow-list', () => {
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
