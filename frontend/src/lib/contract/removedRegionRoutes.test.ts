/**
 * OpenProcessor #104: `PUT /crops/{id}/region` and `PUT /crops/batch_region`
 * were deleted outright (no 410) and the item-level scalar region keys
 * retired; box edits go through `PUT /crops/{id}/regions`. Nothing outside
 * comments may name the removed single-box routes again, in the app source
 * or in the e2e stubs (a stub for a removed route lets a test pass against
 * a wire the backend no longer serves).
 */
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');

function walk(dir: string, exts: string[], out: string[] = []): string[] {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name === 'node_modules' || e.name === '.venv' || e.name.startsWith('.'))
        continue;
      walk(p, exts, out);
    } else if (exts.some((x) => e.name.endsWith(x))) out.push(p);
  }
  return out;
}

// `/crops/<id>/region` as a whole path tail (not region/undo, region_meta,
// region_thumbnail or regions), and `/crops/batch_region` (not batch_regions).
const SINGLE_BOX_ROUTE =
  /\/crops\/(?:\$\{[^}]+\}|\{[^}]+\}|\([^)]*\)|[^/'"`\s]+)\/region(?=['"`?$\s\\)])/;
const BATCH_ROUTE = /\/crops\/batch_region(?![s_])/;

function code(file: string): string[] {
  const lines = readFileSync(file, 'utf8').split('\n');
  const isPy = file.endsWith('.py');
  return lines.filter((l) => {
    const t = l.trim();
    if (t.startsWith('*') || t.startsWith('//') || t.startsWith('/*')) return false;
    if (isPy && (t.startsWith('#') || t.startsWith('"""'))) return false;
    return true;
  });
}

describe('removed single-box region routes (#104)', () => {
  const files = [
    ...walk(path.join(root, 'src'), ['.ts', '.svelte']).filter(
      (f) => !f.endsWith('.test.ts') && !f.endsWith('removedRegionRoutes.test.ts'),
    ),
    ...walk(path.join(root, 'e2e'), ['.py']),
  ];

  it('scans a real file set', () => {
    expect(files.length).toBeGreaterThan(100);
  });

  it('no source or e2e stub references PUT /crops/{id}/region or /crops/batch_region', () => {
    const hits: string[] = [];
    for (const f of files) {
      for (const l of code(f)) {
        if (SINGLE_BOX_ROUTE.test(l) || BATCH_ROUTE.test(l)) {
          hits.push(`${path.relative(root, f)}: ${l.trim()}`);
        }
      }
    }
    expect(hits).toEqual([]);
  });

  it('the pattern itself flags the removed routes and spares the live ones', () => {
    for (const bad of [
      'stub.on("PUT", r"/crops/([^/]+)/region$", h)',
      '`${API_PREFIX}/crops/${id}/region`',
      "'/crops/batch_region'",
    ]) {
      expect(SINGLE_BOX_ROUTE.test(bad) || BATCH_ROUTE.test(bad)).toBe(true);
    }
    for (const ok of [
      'stub.on("PUT", r"/crops/([^/]+)/regions$", h)',
      '`${API_PREFIX}/crops/${id}/region/undo`',
      '`${API_PREFIX}/crops/${id}/region_meta`',
      '`${API_PREFIX}/crops/${id}/region_thumbnail`',
      "'/crops/batch_regions'",
      '/crops/region/undo_batch',
    ]) {
      expect(SINGLE_BOX_ROUTE.test(ok) || BATCH_ROUTE.test(ok)).toBe(false);
    }
  });
});
