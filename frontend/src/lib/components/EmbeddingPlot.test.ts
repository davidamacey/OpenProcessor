/**
 * `<EmbeddingPlot>` is a pointer/canvas-only control (curation-strategy
 * plan §2.7/§5.6) — it must never register a window/document keydown
 * listener, matching the hard constraint CLAUDE.md's Keyboard shortcuts
 * section places on every new control in this phase, and the same
 * constraint `StrategyBar.svelte` already carries (see
 * `StrategyBar.test.ts`'s header comment for why this repo does a static
 * source scan instead of a component-mount assertion: no
 * `@testing-library/svelte` harness exists here).
 *
 * Lasso-select-to-crop-ids math and the color/scale helpers this
 * component uses are pure functions and get their own thorough coverage
 * in `../embeddingPlot.test.ts` — this file only covers what a component-
 * mount test can't: the keydown-listener regression guard, and that the
 * component is wired in lazily (gated behind an `{#if}`, never part of
 * `/clusters`'s initial render) with zero new keydown listeners on that
 * route either.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));

function read(rel: string): string {
  return readFileSync(path.resolve(here, rel), 'utf-8');
}

describe('EmbeddingPlot.svelte', () => {
  it('never calls addEventListener — canvas/pointer-only, zero global listeners', () => {
    const src = read('./EmbeddingPlot.svelte');
    expect(src).not.toMatch(/addEventListener/);
  });

  it('drives the lasso interaction off element-level pointer attributes, not window/document', () => {
    const src = read('./EmbeddingPlot.svelte');
    expect(src).toMatch(/onpointerdown=/);
    expect(src).toMatch(/onpointermove=/);
    expect(src).toMatch(/onpointerup=/);
    expect(src).not.toMatch(/svelte:window/);
    expect(src).not.toMatch(/svelte:document/);
  });

  it('never registers a window/document keydown listener of any kind', () => {
    const src = read('./EmbeddingPlot.svelte');
    expect(src).not.toMatch(/(window|document)\.addEventListener\(['"]keydown['"]/);
    expect(src).not.toMatch(/onkeydown=/);
  });
});

/**
 * Regression coverage for "if I lasso them, the screen only shows the
 * visual, no images to review or select" (2026-09-12 live report). The
 * component's own banner admits the projection is approximate ("don't use
 * it to make final labeling decisions"), yet Assign/Move act on bare
 * lasso-selected ids with nothing to visually confirm first. Fix: a
 * capped thumbnail strip of the selected crops, rendered from `points`
 * (which already carries `crop_id`/`class_name` from the API response) —
 * not a new fetch.
 */
describe('EmbeddingPlot.svelte: selected-crop thumbnail preview', () => {
  const src = read('./EmbeddingPlot.svelte');

  it('renders a thumbnail (getThumbUrl) for the lasso-selected crops, not just the canvas dots', () => {
    expect(src).toMatch(/getThumbUrl\(p\.crop_id/);
  });

  it('derives the preview from already-loaded point data, not a new network fetch', () => {
    const derivedBlock = src.slice(
      src.indexOf('const selectedPreview'),
      src.indexOf(');', src.indexOf('const selectedPreview')) + 2,
    );
    expect(derivedBlock).toMatch(/points\.filter/);
    expect(derivedBlock).not.toMatch(/await|fetch\(/);
  });

  it('caps the rendered preview so a huge lasso cannot render thousands of <img> tags', () => {
    expect(src).toMatch(/SELECTED_PREVIEW_CAP/);
    expect(src).toMatch(/\.slice\(0, SELECTED_PREVIEW_CAP\)/);
    expect(src).toMatch(/\+\{selectedIds\.size - SELECTED_PREVIEW_CAP\} more/);
  });

  it('disables native drag on the preview thumbnails (same Safari fix as CropCard)', () => {
    const start = src.indexOf('{#each selectedPreview');
    const previewImgBlock = src.slice(start, src.indexOf('{/each}', start));
    expect(previewImgBlock).toMatch(/draggable="false"/);
  });
});

describe('/clusters wiring adds zero new keydown listeners', () => {
  it('routes/clusters/+page.svelte still registers zero direct window/document keydown listeners', () => {
    const src = read('../../routes/clusters/+page.svelte');
    expect(src).not.toMatch(/(window|document)\.addEventListener\(['"]keydown['"]/);
  });

  it('the embedding-plot toggle exists and the component is mounted behind an {#if} (lazy, never eager)', () => {
    const src = read('../../routes/clusters/+page.svelte');
    expect(src).toMatch(/EmbeddingPlot/);
    // Lazily mounted: the component tag must be inside a conditional
    // block keyed on the toggle state, not rendered unconditionally at
    // the top of the template.
    expect(src).toMatch(/\{#if\s+showEmbeddingViz\}[\s\S]*<EmbeddingPlot/);
  });

  it('the toggle button is gated on isEmbeddingVizAvailable (absent, not just disabled, for a backend that has not shipped it)', () => {
    const src = read('../../routes/clusters/+page.svelte');
    expect(src).toMatch(/isEmbeddingVizAvailable/);
    expect(src).toMatch(/embeddingVizAvailable/);
  });
});
