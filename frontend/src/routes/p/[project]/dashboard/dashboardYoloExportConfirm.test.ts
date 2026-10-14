/**
 * The dashboard's one-click YOLO export used to run with no confirm and toast
 * "job started" for a synchronous endpoint. Static source scan (no
 * component-mount harness for this page).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('YOLO export requires confirmation and shows the real synchronous result', () => {
  it('the button opens a confirm step, not runExport directly', () => {
    expect(src).toMatch(/onclick=\{openExportConfirm\}/);
    expect(src).not.toMatch(/onclick=\{runExport\}[\s\S]{0,40}Export Dataset/);
  });

  it('runExport never assumes a job_id / "job started" — it renders the served ExportResult', () => {
    const fn = src.match(/async function runExport\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).not.toMatch(/job started/i);
    expect(fn).not.toMatch(/res\.job_id/);
    expect(fn).toMatch(/exportResult = await exportYolo\(\)/);
  });

  it('the confirm modal renders the served export_dir/dataset_sha/split_counts fields', () => {
    const modal = src.match(/\{#if exportConfirmOpen\}[\s\S]*?\n\{\/if\}/)?.[0];
    expect(modal).not.toBeUndefined();
    expect(modal).toMatch(/exportResult\.export_dir/);
    expect(modal).toMatch(/exportResult\.dataset_sha/);
    expect(modal).toMatch(/exportResult\.split_counts/);
    expect(modal).toMatch(/exportError/);
  });
});
