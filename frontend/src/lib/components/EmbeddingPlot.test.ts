/**
 * `<EmbeddingPlot>` is a pointer/canvas-only control: it must never register a window/document
 * keydown listener (the Keyboard shortcuts rule every new control carries). The behaviour is
 * covered by mounting it in `../workspace/views/EmbeddingView.test.ts`; the geometry and colour
 * helpers by `../embeddingPlot.test.ts`; this file keeps the listener guard and the preview's
 * static checks.
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
  const src = read('./EmbeddingPlot.svelte');

  it('never calls addEventListener — canvas/pointer-only, zero global listeners', () => {
    expect(src).not.toMatch(/addEventListener/);
  });

  it('drives the interactions off element-level pointer attributes, not window/document', () => {
    expect(src).toMatch(/onpointerdown=/);
    expect(src).toMatch(/onpointermove=/);
    expect(src).toMatch(/onpointerup=/);
    expect(src).not.toMatch(/svelte:window/);
    expect(src).not.toMatch(/svelte:document/);
  });

  it('never registers a window/document keydown listener of any kind', () => {
    expect(src).not.toMatch(/(window|document)\.addEventListener\(['"]keydown['"]/);
    expect(src).not.toMatch(/onkeydown=/);
  });

  it('renders the legend from the served legend, not from the points', () => {
    expect(src).toContain('data-testid="embedding-legend"');
    expect(src).toContain('{#each legend as e');
  });
});

describe('EmbeddingSelectedPreview.svelte', () => {
  const src = read('./EmbeddingSelectedPreview.svelte');

  it('renders a thumbnail (getThumbUrl) for the selected crops, not just the canvas dots', () => {
    expect(src).toMatch(/getThumbUrl\(p\.crop_id/);
  });

  it('is built from the points the plot holds, with no fetch of its own', () => {
    expect(src).not.toMatch(/await|fetch\(/);
  });

  it('caps the rendered preview so a huge lasso cannot render thousands of <img> tags', () => {
    expect(src).toMatch(/SELECTED_PREVIEW_CAP/);
    expect(src).toMatch(/points\.slice\(0, SELECTED_PREVIEW_CAP\)/);
    expect(src).toMatch(/\+\{count - SELECTED_PREVIEW_CAP\} more/);
  });

  it('disables native drag on the thumbnails (same Safari fix as CropCard)', () => {
    const start = src.indexOf('{#each selectedPreview');
    const block = src.slice(start, src.indexOf('{/each}', start));
    expect(block).toMatch(/draggable="false"/);
  });

  it('scrolls on its own at the plot height and opens a crop enlarged through the shared Modal', () => {
    expect(src).toMatch(/overflow-y-auto/);
    expect(src).toMatch(/style:height="\{height\}px"/);
    expect(src).toMatch(/onclick=\{\(\) => \(expandedPreviewId = p\.crop_id\)\}/);
    // 512 is the backend's hard cap for a crop thumbnail (a larger size is a 422).
    expect(src).toMatch(/getThumbUrl\(expandedPreviewId, 512\)/);
    expect(src).toMatch(/<Modal\s[^>]*label="Selected crop preview"/);
    expect(src).toMatch(/aria-label="Close"/);
    expect(src).not.toMatch(/onkeydown=/);
  });
});
