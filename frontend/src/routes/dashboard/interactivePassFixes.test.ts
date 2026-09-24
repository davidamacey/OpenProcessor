/**
 * Regression tests for the dashboard FRONTEND findings fixed by the
 * 2026-09-24 interactive-pass follow-up (docs/design/
 * interactive-pass-2026-09-24.md §6 FRONTEND): M13 (one-click YOLO
 * export with no confirm, toasting "job started" for a synchronous
 * endpoint) and m6 (hardcoded class-balance legend contradicting the
 * served adequacy thresholds).
 *
 * Same static source-scan convention as logicMovesW3.test.ts — no
 * component-mount harness for this page.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('M13: YOLO export requires confirmation and shows the real synchronous result', () => {
  it('the button opens a confirm step, not runExport directly', () => {
    expect(src).toMatch(/onclick=\{openExportConfirm\}/);
    expect(src).not.toMatch(/onclick=\{runExport\}[\s\S]{0,40}Export Dataset/);
  });

  it('runExport never assumes a job_id / "job started" — it renders the served ExportResult', () => {
    const fn = src.match(/async function runExport\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).not.toMatch(/job started/i);
    expect(fn).not.toMatch(/res\.job_id/);
    expect(fn).toMatch(/exportResult = await exportYolo\(\)/);
  });

  it('the confirm modal renders the served export_dir/dataset_sha/split_counts fields', () => {
    const modal = src.match(/\{#if exportConfirmOpen\}[\s\S]*?\n\{\/if\}/)?.[0];
    expect(modal).toBeDefined();
    expect(modal).toMatch(/exportResult\.export_dir/);
    expect(modal).toMatch(/exportResult\.dataset_sha/);
    expect(modal).toMatch(/exportResult\.split_counts/);
    expect(modal).toMatch(/exportError/);
  });
});

describe('m6: the class-balance legend renders the served adequacy thresholds', () => {
  it('no longer hardcodes 500/100', () => {
    expect(src).not.toMatch(/≥500.*100–499.*<100/);
  });

  it('reads legacyStats.thresholds.block_below/warn_below', () => {
    const legend = src.match(/Class balance \(validated\)<\/h2>[\s\S]*?<\/span>/)?.[0];
    expect(legend).toBeDefined();
    expect(legend).toMatch(/legacyStats\.thresholds\s*\n?\s*\.?warn_below/);
    expect(legend).toMatch(/legacyStats\s*\n?\s*\.thresholds\s*\n?\s*\.block_below/);
  });
});
