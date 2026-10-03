/**
 * Static source scan for T-C3's dataset-export capability gate
 * (docs/design/backend-integration-phase-c-plan-2026-09-20.md §3).
 * This repo has no `@testing-library/svelte` harness (see
 * `TrainForm.test.ts`'s header, `StrategyBar.test.ts`), so — same
 * convention — this asserts the wiring a reviewer would check by eye.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, './+page.svelte'), 'utf-8');

describe('/train dataset-export capability gate', () => {
  it('imports isDatasetExportAvailable from $lib/strategies and datasetExportForSlot from $lib/annotations/datasetExport', () => {
    expect(src).toMatch(
      /import\s*\{[^}]*isDatasetExportAvailable[^}]*\}\s*from\s*['"]\$lib\/strategies['"]/,
    );
    expect(src).toMatch(
      /import\s*\{[^}]*datasetExportForSlot[^}]*\}\s*from\s*['"]\$lib\/annotations\/datasetExport['"]/,
    );
  });

  it('calls strategiesStore.init()', () => {
    expect(src).toMatch(/strategiesStore\.init\(\)/);
  });

  it('derives datasetExportAvailable from strategiesStore.methods.dataset_exports', () => {
    expect(src).toMatch(/const datasetExportAvailable = \$derived\(/);
    expect(src).toMatch(/strategiesStore\.methods\.dataset_exports/);
  });

  // The single highest-value assertion in this file: refreshSingleClassExportStatus()
  // firing unconditionally on mount is the actual 404 this task removes
  // (a status route is not registered on a backend that never advertises
  // that export kind). Scan the
  // onMount(async () => { … }) block specifically, not the whole file —
  // refreshSingleClassExportStatus is still defined and still called, just
  // from the gated $effect further down.
  it('refreshSingleClassExportStatus() is not called inside onMount', () => {
    const onMountMatch = src.match(/onMount\(async \(\) => \{[\s\S]*?\n {2}\}\);/);
    expect(onMountMatch).not.toBeNull();
    expect(onMountMatch?.[0]).not.toMatch(/refreshSingleClassExportStatus/);
  });

  it('the export section and the dataset-kind toggle are both gated on datasetExportAvailable', () => {
    expect(src).toMatch(/\{#if datasetExportAvailable\}/);
    expect(src).toMatch(/\{#if datasetExportAvailable && datasetExportSpec\}/);
  });

  // bakeoff-train-genericization plan §3.2/§6 commit 2: spec.blurb was
  // declared, validated, and rendered nowhere — a hand-written copy sat
  // right above it instead. (That no domain copy comes back is
  // domainNeutral.scan.test.ts's job.)
  it('renders datasetExportSpec.blurb', () => {
    expect(src).toMatch(/datasetExportSpec\.blurb/);
  });

  // v0.4.0 facts: the near-duplicate threshold is the served
  // `dedup_threshold_default` of the export kind, never a literal here.
  // A scan, since the page is too big to mount for one request body.
  it('takes the dedup threshold from the served export entry, with no literal 0.98', () => {
    expect(src).not.toMatch(/0\.98/);
    expect(src).toMatch(/exportDedupDefault\(/);
    expect(src).toMatch(/dedup_threshold: singleClassDedup \? singleClassDedupThreshold/);
    expect(src).toMatch(/\{#if singleClassDedupThreshold != null\}/);
  });
});
