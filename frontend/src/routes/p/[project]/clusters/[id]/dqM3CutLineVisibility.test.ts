/**
 * Regression tests for DQ-M3's frontend half (docs/design/data-quality-
 * pass-2026-09-24.md §7 FRONTEND item 4): the cut line used to assume
 * "crops are already sorted core-first by the API" and drew a line
 * wherever the raw scan first hit a non-core crop, even when most members
 * had a null `cluster_is_core` (class clusters) or the load order wasn't
 * actually core-first (candidate clusters, where core crops resumed
 * after the "boundary"). The behavior contract for computeCutLine() lives
 * in src/lib/clusters/cutLine.test.ts; these pin that the page actually
 * gates the <CutLine> render on its `visible` flag rather than the old
 * `cutLineIndex > 0 && cutLineIndex < group.items.length` guess.
 *
 * Same static source-scan convention as the other clusters/[id] logic-
 * moves tests — no `@testing-library/svelte` mount harness for a page
 * this size.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('DQ-M3: <CutLine> render is gated on the computed visible flag', () => {
  it('the template checks cutLineVisible, not just an index-bounds guess', () => {
    expect(src).toMatch(
      /\{#if !groupBySubcluster && cutLineVisible && i === cutLineIndex\}/,
    );
    expect(src).not.toMatch(/cutLineIndex > 0 && cutLineIndex < group\.items\.length/);
  });

  it('cutLineVisible is derived from computeCutLine, not hardcoded true', () => {
    expect(src).toMatch(/const cutLineVisible = \$derived\(cutLine\.visible\);/);
  });
});
