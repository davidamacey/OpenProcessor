/**
 * DQ-M8 (docs/design/data-quality-pass-2026-09-24.md §7 FRONTEND item 5):
 * the review panel's repro was exactly "Current label widget_a (vlm)"
 * directly above "Confidence 94.6%" — `label_confidence` is the
 * classifier-detector score on every row (including VLM-sourced ones,
 * range 0.25-0.97 per the audit), never the VLM's own confidence, but the
 * row was unconditionally labeled "Confidence". Fixed to relabel
 * "Detector score" whenever the current label's role (served, via
 * classSourcesStore) is VLM-sourced, with the VLM's own categorical
 * confidence (`vlm_confidence`) as its own row.
 *
 * Same static source-scan convention as the other review/+page.svelte
 * regression tests — no mount harness for a page this size.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('DQ-M8: the review panel labels the detector score for what it is', () => {
  it('isCurrentLabelVlmSourced reads a served role, not a hardcoded string match', () => {
    expect(src).toMatch(
      /const isCurrentLabelVlmSourced = \$derived\(\s*\(classSourcesStore\.roleFor\(current\?\.label_source\) \?\? ''\)\.startsWith\('vlm'\),?\s*\);/,
    );
  });

  it('the row label switches to "Detector score" for a VLM-sourced label', () => {
    expect(src).toMatch(
      /\{isCurrentLabelVlmSourced \? 'Detector score' : 'Confidence'\}/,
    );
  });

  it("the VLM's own categorical confidence renders as its own row when present", () => {
    const idx = src.indexOf('{#if current.vlm_confidence}');
    expect(idx).toBeGreaterThan(-1);
    const slice = src.slice(idx, idx + 150);
    expect(slice).toMatch(/VLM confidence/);
  });
});

/**
 * dq-queues cutover (2026-09-24): class_confidence/vlm_raw_class/
 * vlm_class_empty_reason fold into EXISTING rows (Current label,
 * Confidence/Detector score, VLM confidence) rather than getting their
 * own dt/dd pairs — a real live regression (e2e's
 * test_review_crop_viewport.py) surfaced when they first landed as three
 * new rows: with every field present (a real VLM-labeled item commonly
 * carries all of them), the panel's total height grew enough to push
 * Confirm/Skip/Discard below the fold at 1280x720, regressing DQ-M5 (the
 * review crop panel is height-budgeted via max-h-[46%] specifically so
 * those buttons stay visible without scrolling). No new dt/dd row means
 * no regression regardless of how many of these fields a given item
 * carries at once.
 */
describe('dq-queues cutover: class_confidence/vlm_raw_class/vlm_class_empty_reason fold into existing rows (DQ-M5 height budget)', () => {
  it('vlm_raw_class/vlm_class_empty_reason render inside the Current label dd, not their own dt/dd pair', () => {
    const idx = src.indexOf('<dt class="text-zinc-500">Current label</dt>');
    expect(idx).toBeGreaterThan(-1);
    const ddEnd = src.indexOf('</dd>', idx);
    const block = src.slice(idx, ddEnd);
    expect(block).toMatch(/current\.vlm_raw_class/);
    expect(block).toMatch(/current\.vlm_class_empty_reason/);
    // No separate dt for either — they're inline spans inside this dd.
    expect(src).not.toMatch(/<dt class="text-zinc-500">VLM said<\/dt>/);
    expect(src).not.toMatch(/<dt class="text-zinc-500">VLM empty reason<\/dt>/);
  });

  it('class_confidence folds into the Confidence/Detector score row and the VLM confidence row, never its own "Label confidence" row', () => {
    expect(src).not.toMatch(/<dt class="text-zinc-500">Label confidence<\/dt>/);
    expect(src).toMatch(/current\.class_confidence != null && !current\.vlm_confidence/);
    const vlmConfIdx = src.indexOf('{#if current.vlm_confidence}');
    const vlmConfSlice = src.slice(vlmConfIdx, vlmConfIdx + 700);
    expect(vlmConfSlice).toMatch(/current\.class_confidence/);
  });
});
