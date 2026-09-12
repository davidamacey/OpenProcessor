/**
 * Static-source-scan regression guard for the cross-origin plate-thumbnail
 * bug (see `plateThumbUrl.test.ts` for the executable helper coverage).
 * The actual bug wasn't the missing helper — it was call sites building
 * `/curation/crops/{id}/plate_thumbnail` as a raw string instead of routing
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

const RAW_PLATE_THUMB_PATTERN = /['"`]\/curation\/crops\/.*?plate_thumbnail/;

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
  if (rel.endsWith('.test.ts') || rel.endsWith('.svelte-kit')) return true;
  return false;
}

describe('no raw plate_thumbnail URL construction outside the api.ts helpers', () => {
  const files = walk(srcRoot).filter(
    (f) => (f.endsWith('.ts') || f.endsWith('.svelte')) && !isExcluded(f),
  );

  it('scans at least the known call-site files (sanity check the walk works)', () => {
    const rels = files.map((f) => path.relative(srcRoot, f));
    expect(rels).toContain(path.join('lib', 'components', 'PlateCard.svelte'));
    expect(rels.some((r) => r.includes('clusters') && r.endsWith('+page.svelte'))).toBe(
      true,
    );
  });

  for (const file of files) {
    const rel = path.relative(srcRoot, file);
    it(`${rel} builds no raw /curation/.../plate_thumbnail template string`, () => {
      const src = readFileSync(file, 'utf-8');
      expect(src).not.toMatch(RAW_PLATE_THUMB_PATTERN);
    });
  }
});

describe('PlateCard.svelte uses the shared helpers, not a bare fallback string', () => {
  const src = readFileSync(
    path.resolve(libRoot, 'components/PlateCard.svelte'),
    'utf-8',
  );

  it('imports getPlateThumbUrl and resolveApiUrl from $lib/api', () => {
    expect(src).toMatch(/getPlateThumbUrl/);
    expect(src).toMatch(/resolveApiUrl/);
  });

  it('resolves the server-supplied plate_thumbnail_url through resolveApiUrl, not verbatim', () => {
    expect(src).toMatch(/resolveApiUrl\(crop\.plate_thumbnail_url\)/);
  });

  it('falls back to getPlateThumbUrl(crop.crop_id), never a bare template string', () => {
    expect(src).toMatch(/getPlateThumbUrl\(crop\.crop_id\)/);
    expect(src).not.toMatch(RAW_PLATE_THUMB_PATTERN);
  });
});
