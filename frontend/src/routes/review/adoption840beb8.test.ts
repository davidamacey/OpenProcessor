import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

/**
 * OpenProcessor main 840beb8 adoption: rejection_reasons vocabulary,
 * region_bbox_correct, and the generic served-enum review filter bar
 * (`ReviewFilterSpec`, e.g. Regions' `region_status`). Static source-scan
 * (no @testing-library/svelte in this repo — see reviewFilterConsistency
 * .test.ts's header comment for the established precedent).
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, './+page.svelte'), 'utf-8');

describe('840beb8: generic served-enum filter bar (no tab/param-specific code)', () => {
  it('renders one <select> per activeFilterSpecs entry, keyed by spec.param', () => {
    expect(src).toMatch(/\{#each activeFilterSpecs as spec \(spec\.param\)\}/);
    expect(src).toMatch(
      /onchange=\{\(e\) => setEnumFilter\(spec\.param, e\.currentTarget\.value\)\}/,
    );
  });

  it('activeFilterSpecs is derived from reviewTabsVocabularyStore.filterSpecsFor(activeTabEndpointId)', () => {
    expect(src).toMatch(
      /const activeFilterSpecs = \$derived\(\s*reviewTabsVocabularyStore\.filterSpecsFor\(activeTabEndpointId\),?\s*\);/,
    );
  });

  it('_filter() forwards only params the active tab declares in filter_specs', () => {
    const filterFnStart = src.indexOf('function _filter()');
    const filterFnBody = src.slice(filterFnStart, src.indexOf('\n  }\n', filterFnStart));
    expect(filterFnBody).toMatch(/Object\.assign\(f, activeEnumParams\);/);
    // enumFilterValues is seeded from every URL param, so iterating it
    // directly would forward unrelated/stale params to the backend.
    expect(filterFnBody).not.toMatch(/enumFilterValues/);
    expect(src).toMatch(
      /for \(const spec of activeFilterSpecs\) \{\s*const value = enumFilterValues\[spec\.param\];\s*if \(value\) out\[spec\.param\] = value;\s*\}/,
    );
  });

  it('setEnumFilter persists the value in the URL (like preset)', () => {
    const fnStart = src.indexOf('function setEnumFilter(');
    expect(fnStart).toBeGreaterThan(-1);
    const fnBody = src.slice(fnStart, src.indexOf('\n  }\n', fnStart));
    expect(fnBody).toMatch(/url\.searchParams\.set\(param, value\)/);
    expect(fnBody).toMatch(/url\.searchParams\.delete\(param\)/);
    expect(fnBody).toMatch(/replaceState\(url, \{\}\)/);
  });

  it('enumFilterValues resets (state + URL) on every tab-click, not just on mount', () => {
    const idx = src.indexOf("url.searchParams.set('tab', t.urlId);");
    expect(idx).toBeGreaterThan(-1);
    const nearby = src.slice(idx, idx + 500);
    expect(nearby).toMatch(/for \(const param of Object\.keys\(enumFilterValues\)\)/);
    expect(nearby).toMatch(/enumFilterValues = \{\};/);
  });

  it('the debounced filter effect keys on activeEnumParams, so a select change or a late-loading spec refetches', () => {
    // Keying on the raw enumFilterValues missed the case where a
    // URL-seeded ?region_status= was set before /review/tabs loaded: the
    // spec arriving changes what _filter() sends but not enumFilterValues.
    // e2e test_region_status_from_the_url_reaches_the_queue_request
    // covers the behavior; this pins the wiring.
    const effectIdx = src.indexOf('let lastFilterKey: string | null = null;');
    expect(effectIdx).toBeGreaterThan(-1);
    const effectBody = src.slice(effectIdx, effectIdx + 1500);
    expect(effectBody).toMatch(/void activeEnumParams;/);
    expect(effectBody).toMatch(/activeEnumParams,\s*\]\);/);
    expect(effectBody).not.toMatch(/void enumFilterValues;/);
  });
});

describe('840beb8: region_bbox_correct folded into the existing Validation row', () => {
  it('the inline slot panel renders a "model: box wrong" chip gated on boxCorrect === false', () => {
    expect(src).toMatch(/slotData\?\.lifecycle\?\.boxCorrect === false/);
    expect(src).toMatch(/model: box wrong/);
  });
});

describe('840beb8: per-item reason wording — region_rejection_reason wins over the generic reason string', () => {
  it('currentSlotRejectionReason is derived off slotOf(current, activeSlot).lifecycle.rejectionReason', () => {
    expect(src).toMatch(/const currentSlotRejectionReason = \$derived<string \| null>\(/);
    expect(src).toMatch(
      /slotOf\(current, activeSlot\)\?\.lifecycle\?\.rejectionReason \?\? null/,
    );
  });

  it('the Reason row falls back to current.reason ONLY when there is no currentSlotRejectionReason', () => {
    const idx = src.indexOf('{#if currentSlotRejectionReason}');
    expect(idx).toBeGreaterThan(-1);
    const block = src.slice(idx, src.indexOf('{/if}', idx) + 5);
    expect(block).toMatch(/\{:else\}/);
    expect(block).toMatch(/current\.reason \?\? '—'/);
    // The rejection branch must use the vocabulary label, never the raw
    // per-item `reason` string (which always says "verifier rejected
    // this candidate (…)", wrong wording for a needs_human item).
    const rejectionBranch = block.slice(0, block.indexOf('{:else}'));
    expect(rejectionBranch).toMatch(/regionVocabularyStore\.rejectionReasonLabel\(/);
    expect(rejectionBranch).not.toMatch(/current\.reason/);
  });

  it("a needs_human item's Reason-row label is never worded as a rejection", () => {
    const idx = src.indexOf('{#if currentSlotRejectionReason}');
    const block = src.slice(idx, src.indexOf('{:else}', idx));
    expect(block).toMatch(/reasonKind === 'needs_human' \? 'Needs review' : 'Rejection'/);
  });
});

describe('840beb8: candidate badges style by the served rejection kind (never "rejected" for needs_human)', () => {
  it('the inline slot-panel candidate badge branches on rejectionReasonKind, not just presence of a reason', () => {
    expect(src).toMatch(
      /\{@const candidateKind = regionVocabularyStore\.rejectionReasonKind\(\s*slotData\?\.lifecycle\?\.rejectionReason,?\s*\)\}/,
    );
    expect(src).toMatch(
      /candidateKind === 'needs_human'\s*\?\s*'candidate · needs review'\s*:\s*'rejected candidate · confirm to accept'/,
    );
  });
});
