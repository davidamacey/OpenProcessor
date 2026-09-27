import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

/**
 * Regression test for the stale /export page cleanup (2026-09): the page
 * used to render a "Training command" panel telling the operator to run
 * a `train_medium.sh` shell script — a script
 * path that no longer exists, predating the real `/train` cockpit that now
 * fully replaces this workflow. This is a static source-scan (no
 * @testing-library/svelte in this repo — see clusterMoveRace.test.ts /
 * reviewFilterConsistency.test.ts for the established precedent) proving:
 * the stale panel is gone, a `/train` link points the operator to the real
 * next step, and the registry download buttons are gated on the served
 * `GET {API_PREFIX}/export/datasets` availability flag (m15, 2026-09-24
 * interactive pass — see exportDatasetRows.test.ts for the behavioral
 * coverage of the gating logic itself, `hasCurrentMulticlassExport()`).
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, './+page.svelte'), 'utf-8');

describe('export page: stale training-command panel removed', () => {
  it('no longer references the dead train_medium.sh script path', () => {
    expect(src).not.toMatch(/train_medium\.sh/);
  });

  it('has no leftover "Training command" heading', () => {
    expect(src).not.toMatch(/Training command/);
  });

  it('has no leftover copyCommand/trainingCommand helpers', () => {
    expect(src).not.toMatch(/trainingCommand/);
    expect(src).not.toMatch(/copyCommand/);
    expect(src).not.toMatch(/totalImagesTraining/);
  });

  it('links the operator to /train as the next step', () => {
    expect(src).toMatch(/<a href=\{resolve\(projectHref\('\/train'\)\)\}/);
  });
});

describe('export page: registry download buttons gated on the served dataset list (m15)', () => {
  function buttonBlock(marker: string): string {
    const idx = src.indexOf(marker);
    expect(idx).toBeGreaterThan(-1);
    const start = src.lastIndexOf('<button', idx);
    const end = src.indexOf('</button>', idx);
    return src.slice(start, end);
  }

  it('class_registry.json button is disabled without a current multi-class export', () => {
    const block = buttonBlock("getClassRegistryUrl(), 'class_registry.json'");
    expect(block).toMatch(/disabled=\{!hasMulticlassExport\}/);
  });

  it('data.yaml button is disabled without a current multi-class export', () => {
    const block = buttonBlock("getDataYamlUrl(), 'data.yaml'");
    expect(block).toMatch(/disabled=\{!hasMulticlassExport\}/);
  });

  it('manifest.json download button (registry row) is disabled without a current multi-class export', () => {
    const block = buttonBlock("getManifestUrl(), 'manifest.json'");
    expect(block).toMatch(/disabled=\{!hasMulticlassExport\}/);
  });

  it('no longer gates any of the three on the shared export-job-status slot', () => {
    expect(src).not.toMatch(/disabled=\{exportState\?\.status !== 'success'\}/);
  });
});
