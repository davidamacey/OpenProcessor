/**
 * W1 (docs/design/logic-moves-adoption-plan-2026-09-24.md) — /clusters/[id]'s
 * D (discard) and reject-VLM actions moved onto the backend's discard/
 * vlm_dismiss endpoints. This repo has no component-mount harness (see
 * clusterMoveRace.test.ts's doc comment); pinned via the same static
 * source-scan convention.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('W1: D (discard) calls the discard endpoints, not DELETE /label', () => {
  it('never imports the deleted deleteCropLabel', () => {
    expect(src).not.toMatch(/deleteCropLabel/);
  });

  it('single selection calls discardCrop; multi selection calls discardCropsBatch', () => {
    const handler = src.match(/reg\(\s*'d',([\s\S]*?)'Discard selected',/)?.[0];
    expect(handler).toBeDefined();
    expect(handler).toMatch(/await discardCrop\(ids\[0\]!\)/);
    expect(handler).toMatch(/await discardCropsBatch\(ids\)/);
  });

  it('records an undo entry for exactly the server-confirmed ids', () => {
    const handler = src.match(/reg\(\s*'d',([\s\S]*?)'Discard selected',/)?.[0];
    expect(handler).toMatch(/undoStore\.recordWrites\(succeededIds\)/);
  });
});

describe('W1: reject-VLM calls vlm_dismiss and renders the returned item', () => {
  it('rejectVlmForCrop no longer clears the suggestion locally', () => {
    const fn = src.match(/async function rejectVlmForCrop\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/await vlmDismissCrop\(crop\.id\)/);
    expect(fn).toMatch(
      /cropPager\.items = cropPager\.items\.map\(\(c\) => \(c\.id === crop\.id \? item : c\)\)/,
    );
  });

  it('a 409 (no suggestion) is shown as an info toast, not an error', () => {
    const fn = src.match(/async function rejectVlmForCrop\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toMatch(/e\.status === 409/);
    expect(fn).toMatch(/toastStore\.info\(/);
  });
});

describe('W1: bulk label writes record undo off the server-served updated_ids', () => {
  it('every recordWrites call site reads res.updated_ids, never a locally-filtered id list', () => {
    expect(src).not.toMatch(/recordWrites\(ids, res\.conflicts/);
    const bulkLabelSites = [
      ...src.matchAll(
        /await bulkLabel\([^)]*\);\n\s*undoStore\.recordWrites\(([^)]*)\)/g,
      ),
    ];
    expect(bulkLabelSites.length).toBeGreaterThan(0);
    for (const m of bulkLabelSites) {
      expect(m[1]).toBe('res.updated_ids');
    }
  });
});
