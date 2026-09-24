/**
 * DQ-M8 (docs/design/data-quality-pass-2026-09-24.md §7 FRONTEND item 5):
 * the review panel's repro was exactly "Current label dumptruck (vlm)"
 * directly above "Confidence 94.6%" — `label_confidence` is the
 * vehicle-detector/v6 score on every row (including VLM-sourced ones,
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
