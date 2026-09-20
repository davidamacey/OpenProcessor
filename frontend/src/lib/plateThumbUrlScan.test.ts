/**
 * Static-source-scan regression guard for the cross-origin plate-thumbnail
 * bug (see `plateThumbUrl.test.ts` for the executable helper coverage).
 * The actual bug wasn't the missing helper — it was call sites building
 * `{API_PREFIX}/crops/{id}/region_thumbnail` as a raw string instead of routing
 * through it. This repo has no `@testing-library/svelte` harness, so a
 * static scan (the convention `EmbeddingPlot.test.ts` and
 * `StrategyBar.test.ts` already use for this kind of "never do X again"
 * guard) is the only mount-free way to pin the call sites.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const libRoot = here;
const srcRoot = path.resolve(here, '..');

/**
 * Matches a hand-built region-thumbnail URL: anything that opens a string
 * literal or closes a `${…}` expression and then walks a
 * `/crops/…/{plate,region}_thumbnail` path, with or without a literal
 * prefix segment in between.
 *
 * Deliberately prefix-agnostic. Before `API_PREFIX` a rogue call site
 * looked like `'/curation/crops/…'`; after it, like
 * `` `${apiBase}${API_PREFIX}/crops/…` ``. A `/curation`-literal regex catches
 * the first and silently misses the second, which is the exact way this
 * kind of guard rots.
 *
 * Deliberately segment-agnostic too. `region_thumbnail` is the real
 * route; `plate_thumbnail` is the segment T-B2 removed, which 404s and
 * which the backend will never alias
 * (cropwright_backend_integration_plan.md §3.2). Matching BOTH means this
 * guard catches a hand-rolled URL whichever name a future call site
 * reaches for — and catches a revert to the dead one anywhere.
 */
const RAW_REGION_THUMB_PATTERN =
  /(?:['"`]|\})(?:\/[a-z_]+)?\/crops\/[^'"`]*?(?:plate|region)_thumbnail/;

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      walk(full, out);
    } else {
      out.push(full);
    }
  }
  return out;
}

function isExcluded(file: string): boolean {
  const rel = path.relative(srcRoot, file);
  if (rel === path.join('lib', 'api.ts')) return true;
  // Slot profiles declare prefix-RELATIVE path templates by design
  // (`/crops/{id}/region_thumbnail`, joined with API_PREFIX at the call
  // site — see annotations/cohorts.ts:36). That is the sanctioned
  // declaration point, not a hand-rolled fetch URL.
  if (rel.startsWith(path.join('lib', 'annotations', 'profiles') + path.sep)) return true;
  if (rel.endsWith('.test.ts') || rel.endsWith('.svelte-kit')) return true;
  return false;
}

describe('no raw region_thumbnail URL construction outside the api.ts helpers', () => {
  const files = walk(srcRoot).filter(
    (f) => (f.endsWith('.ts') || f.endsWith('.svelte')) && !isExcluded(f),
  );

  it('scans at least the known call-site files (sanity check the walk works)', () => {
    const rels = files.map((f) => path.relative(srcRoot, f));
    expect(rels).toContain(path.join('lib', 'components', 'SlotCard.svelte'));
    expect(rels.some((r) => r.includes('clusters') && r.endsWith('+page.svelte'))).toBe(
      true,
    );
  });

  for (const file of files) {
    const rel = path.relative(srcRoot, file);
    it(`${rel} builds no raw /crops/.../{plate,region}_thumbnail template string`, () => {
      const src = readFileSync(file, 'utf-8');
      expect(src).not.toMatch(RAW_REGION_THUMB_PATTERN);
    });
  }
});

describe('SlotCard.svelte uses the shared helpers, not a bare fallback string', () => {
  const src = readFileSync(path.resolve(libRoot, 'components/SlotCard.svelte'), 'utf-8');

  it('imports getRegionThumbUrl and resolveApiUrl from $lib/api', () => {
    expect(src).toMatch(/getRegionThumbUrl/);
    expect(src).toMatch(/resolveApiUrl/);
  });

  it('resolves the server-supplied plate_thumbnail_url through resolveApiUrl, not verbatim', () => {
    expect(src).toMatch(/resolveApiUrl\(crop\.plate_thumbnail_url\)/);
  });

  it('falls back to getRegionThumbUrl(crop.crop_id), never a bare template string', () => {
    expect(src).toMatch(/getRegionThumbUrl\(crop\.crop_id\)/);
    expect(src).not.toMatch(RAW_REGION_THUMB_PATTERN);
  });
});

/**
 * `isExcluded()` skips `lib/annotations/profiles/` — profiles declare
 * prefix-relative path templates by design, so the scan above cannot
 * distinguish a sanctioned declaration from a rogue one. That exclusion
 * is what let the profile keep a `plate_thumbnail` template pointing at
 * an unregistered route while every scanned file was clean. Pin the one
 * live profile's segment explicitly instead.
 */
describe('the live license_plate profile declares the registered route segment', () => {
  const src = readFileSync(
    path.resolve(libRoot, 'annotations/profiles/licensePlate.ts'),
    'utf-8',
  );

  it('uses region_thumbnail, the segment the backend actually registers', () => {
    expect(src).toMatch(/\/crops\/\$\{encodeURIComponent\(id\)\}\/region_thumbnail/);
  });

  it('no longer declares the dead plate_thumbnail segment', () => {
    expect(src).not.toMatch(/\/plate_thumbnail/);
  });
});
