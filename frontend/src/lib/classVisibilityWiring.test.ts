/**
 * Static source-scan regression test for the widget_tag assignment-hiding
 * feature. This repo has no @testing-library/svelte (see clusterMoveRace.test.ts
 * / EmbeddingPlot.test.ts for precedent), so wiring correctness is verified by
 * grepping the actual .svelte/.ts sources rather than mounting components.
 *
 * Covers:
 *  (a) every class-ASSIGNMENT surface imports $lib/classVisibility
 *  (b) no raw unfiltered `{#each classesStore.classes as cls` remains inside
 *      a <select> on the cluster-detail page or the embedding plot
 *  (c) the class-SUBSET picker (export/train) does NOT import classVisibility
 *      — an over-application guard, since that surface must keep offering
 *      widget_tag for export/training per the task's explicit "do NOT
 *      touch" list
 *  (d) the review page's class FILTER dropdown logic is unchanged — it still
 *      filters only on `!c.deprecated`, not through classVisibility
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..', '..');

function read(rel: string): string {
  return readFileSync(path.resolve(root, rel), 'utf-8');
}

const ASSIGNMENT_SURFACES = [
  'src/lib/classPicker.ts',
  'src/lib/stores/classes.svelte.ts',
  'src/lib/components/ClassSidebar.svelte',
  'src/routes/clusters/[id]/+page.svelte',
  'src/lib/components/EmbeddingPlot.svelte',
  'src/lib/components/ShortcutOverlay.svelte',
  'src/lib/classHotkey.ts',
  'src/routes/+layout.svelte',
];

describe('widget_tag assignment-hiding wiring', () => {
  it.each(ASSIGNMENT_SURFACES)('%s imports from $lib/classVisibility', (rel) => {
    const src = read(rel);
    expect(src).toMatch(/from ['"]\$lib\/classVisibility['"]/);
  });

  it('cluster-detail confirm-to <select> filters through isAssignableClass, not raw classesStore.classes', () => {
    const src = read('src/routes/clusters/[id]/+page.svelte');
    expect(src).not.toMatch(/\{#each classesStore\.classes as cls/);
    expect(src).toMatch(
      /\{#each classesStore\.classes\.filter\(isAssignableClass\) as cls/,
    );
  });

  it('embedding-plot assign <select> filters through isAssignableClass, not raw classesStore.classes', () => {
    const src = read('src/lib/components/EmbeddingPlot.svelte');
    expect(src).not.toMatch(/\{#each classesStore\.classes as cls/);
    expect(src).toMatch(
      /\{#each classesStore\.classes\.filter\(isAssignableClass\) as cls/,
    );
  });

  it('the class-subset picker (export/train) does NOT import classVisibility', () => {
    const src = read('src/lib/components/ClassSubsetPicker.svelte');
    expect(src).not.toMatch(/classVisibility/);
  });

  it("the review page's class FILTER dropdown pool is unchanged (still !c.deprecated only)", () => {
    const src = read('src/routes/review/+page.svelte');
    expect(src).toMatch(
      /const filterableClasses = \$derived\(classesStore\.classes\.filter\(\(c\) => !c\.deprecated\)\);/,
    );
  });

  it('the review page does not import classVisibility itself — it delegates to classPicker.ts/classesStore', () => {
    const src = read('src/routes/review/+page.svelte');
    expect(src).not.toMatch(/classVisibility/);
  });
});
