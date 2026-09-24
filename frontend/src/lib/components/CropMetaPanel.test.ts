/**
 * `<CropMetaPanel>` (the cluster ⓘ detail modal's body). No
 * `@testing-library/svelte` mount harness in this repo (see
 * ClusterBadge.test.ts's header comment) — static source scan.
 *
 * Covers two frontend fixes from docs/design/data-quality-pass-2026-09-24.md:
 *
 *  - DQ-m7: the modal showed item text and class history but never the
 *    plate's status/text/candidates/chain, because it looked up slots via
 *    `forClass(crop.class_id, ...)` — which only ever matches a crop
 *    literally classified "license_plate", never the vehicle crops (class
 *    "sedan", "suv", ...) a plate sub-box actually lives on. Fixed to
 *    iterate every REGISTERED slot and gate on slotIsPresent(), the same
 *    presence check every other slot-generic surface in this app uses.
 *  - DQ-M8: `label_confidence` is the detector/v6 score on every row
 *    (including VLM-sourced ones), but the row was unconditionally
 *    labeled "Confidence" next to a VLM-sourced label, reading as the
 *    VLM's own certainty. Now relabeled "Detector score" for a
 *    VLM-sourced label (served role, not a hardcoded string check), with
 *    the VLM's own categorical confidence (`vlm_confidence`) as its own
 *    row.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, 'CropMetaPanel.svelte'), 'utf-8');

describe('DQ-m7: plate/slot fields render for any crop carrying slot evidence, not just class-bound ones', () => {
  it("iterates every registered slot (slotRegistry.all), not slots bound to the crop's own class", () => {
    expect(src).toMatch(/const presentSlots = \$derived\(\s*slotRegistry\.all\.filter/);
    // The old `boundSlots` derived (backed by forClass()) is gone — a doc
    // comment above still mentions the old call for context.
    expect(src).not.toMatch(/const boundSlots = /);
  });

  it('gates each slot block on slotIsPresent(slotOf(crop, spec)) — real evidence, not class binding', () => {
    expect(src).toMatch(
      /slotRegistry\.all\.filter\(\(spec\) => slotIsPresent\(slotOf\(crop, spec\)\)\)/,
    );
    expect(src).toMatch(/\{#each presentSlots as spec \(spec\.key\)\}/);
  });
});

describe('DQ-M8: the detector score is labeled for what it is next to a VLM-sourced label', () => {
  it('the row label is conditional on a served role, not a hardcoded class_source/label_source string match', () => {
    expect(src).toMatch(
      /const isVlmSourced = \$derived\(\s*\(classSourcesStore\.roleFor\(classSource\) \?\? ''\)\.startsWith\('vlm'\),?\s*\);/,
    );
    expect(src).not.toMatch(/classSource === 'vlm'/);
    expect(src).not.toMatch(/classSource\.startsWith\('vlm'\)/);
  });

  it('renders "Detector score" when VLM-sourced, plain "Confidence" otherwise', () => {
    expect(src).toMatch(
      /<dt class="text-zinc-500">\{isVlmSourced \? 'Detector score' : 'Confidence'\}<\/dt>/,
    );
  });

  it("the VLM's own categorical confidence gets its own row, not folded into the score row", () => {
    const idx = src.indexOf('{#if vlmConf}');
    expect(idx).toBeGreaterThan(-1);
    const slice = src.slice(idx, idx + 150);
    expect(slice).toMatch(/VLM confidence/);
  });
});

describe('dq-queues cutover (2026-09-24): class_confidence / vlm_raw_class / vlm_class_empty_reason', () => {
  it('renders a "Label confidence" row gated on class_confidence != null, with the source prefix', () => {
    expect(src).toMatch(/\{#if crop\.class_confidence != null\}/);
    expect(src).toMatch(/Label confidence/);
    expect(src).toMatch(/classConfidenceSourceLabel/);
  });

  it("shows the VLM's verbatim class answer when present", () => {
    expect(src).toMatch(/\{#if crop\.vlm_raw_class\}/);
    expect(src).toMatch(/VLM said/);
  });

  it('shows vlm_class_empty_reason when set', () => {
    expect(src).toMatch(/\{#if crop\.vlm_class_empty_reason\}/);
    expect(src).toMatch(/VLM empty reason/);
  });
});
