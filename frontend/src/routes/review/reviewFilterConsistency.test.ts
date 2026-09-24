import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

/**
 * Regression test for the "tabs are inconsistent" bug (2026-09): the
 * SubjectScopeToggle (rank-scope) and BlurSlider (clarity) controls used to
 * be gated behind `{#if PRIMARY_TABS.includes(effectiveTab)}`, so they
 * appeared only on the Primary·Low-Conf preset chip and the COCO Blind Spots
 * tab and silently vanished everywhere else (All, Uncertainty, Model
 * Disagreements, Regions, Gemma-mismatches chip, Gemma-low-conf chip) — even
 * though the backend (`review.py`) always treats `max_rank` /
 * `min_blur_ratio` as tab-agnostic ("Both apply across tabs").
 *
 * dq-queues cutover (2026-09-24): `GET {API_PREFIX}/review/tabs` now serves
 * each tab's own `filters` list, so every filter-bar control (including
 * these two) is gated again — but on `filterVisible(param)`
 * (reviewTabsVocabularyStore.filterSupported, keyed off the served list,
 * defaulting to visible when unknown), never on a hardcoded
 * PRIMARY_TABS/effectiveTab tab-id list. This is a static source-scan (no
 * @testing-library/svelte in this repo — see clusterMoveRace.test.ts for
 * the established precedent) proving the gate is the served-filter one and
 * these controls/filters are tab-list-agnostic like Conf/Class/HDD
 * source.
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, './+page.svelte'), 'utf-8');

describe('review page: rank-scope + blur controls are tab-agnostic (no PRIMARY_TABS gate)', () => {
  it('never references PRIMARY_TABS anywhere in the page', () => {
    expect(src).not.toMatch(/PRIMARY_TABS/);
  });

  it('does not import PRIMARY_TABS from reviewTabs', () => {
    expect(src).not.toMatch(
      /import\s*\{[^}]*PRIMARY_TABS[^}]*\}\s*from\s*['"]\$lib\/reviewTabs['"]/,
    );
  });

  it('renders SubjectScopeToggle with no #if wrapper gating it to specific tabs', () => {
    const idx = src.indexOf('<SubjectScopeToggle');
    expect(idx).toBeGreaterThan(-1);
    // Walk backward from the component tag to the nearest control-flow
    // block opener; it must not be an {#if ...includes(effectiveTab)...}
    // guard (or any {#if} at all immediately wrapping just these controls).
    const preceding = src.slice(0, idx);
    const lastIfMatch = [...preceding.matchAll(/\{#if [^}]*\}/g)].pop();
    if (lastIfMatch) {
      expect(lastIfMatch[0]).not.toMatch(/includes\(effectiveTab\)/);
      expect(lastIfMatch[0]).not.toMatch(/PRIMARY_TABS/);
    }
  });

  it('gates SubjectScopeToggle/BlurSlider on filterVisible(...), never on PRIMARY_TABS/effectiveTab', () => {
    const scopeIdx = src.indexOf('<SubjectScopeToggle');
    const blurIdx = src.indexOf('<BlurSlider', scopeIdx);
    expect(blurIdx).toBeGreaterThan(scopeIdx);
    const between = src.slice(scopeIdx, blurIdx);
    // A gate may sit between them now (each control has its own served-
    // filter check), but it must be the filterVisible one, not a
    // hardcoded tab-id list.
    expect(between).not.toMatch(/PRIMARY_TABS/);
    expect(between).not.toMatch(/includes\(effectiveTab\)/);

    const scopeBlock = src.slice(src.lastIndexOf('{#if', scopeIdx), scopeIdx);
    expect(scopeBlock).toMatch(/filterVisible\('max_rank'\)/);
    const blurBlock = src.slice(src.lastIndexOf('{#if', blurIdx), blurIdx);
    expect(blurBlock).toMatch(/filterVisible\('min_blur_ratio'\)/);
  });

  it('applies max_rank whenever subjectScope is set, regardless of tab (no PRIMARY_TABS.includes guard in _filter)', () => {
    const filterFnStart = src.indexOf('function _filter()');
    const filterFnBody = src.slice(filterFnStart, src.indexOf('\n  }\n', filterFnStart));
    expect(filterFnBody).toMatch(/if \(subjectScope !== 0\) f\.max_rank = subjectScope;/);
    expect(filterFnBody).not.toMatch(/PRIMARY_TABS/);
    expect(filterFnBody).not.toMatch(/includes\(effectiveTab\)/);
  });

  it('applies min_blur_ratio whenever minBlurRatio is set, regardless of tab', () => {
    const filterFnStart = src.indexOf('function _filter()');
    const filterFnBody = src.slice(filterFnStart, src.indexOf('\n  }\n', filterFnStart));
    expect(filterFnBody).toMatch(
      /if \(minBlurRatio != null\) f\.min_blur_ratio = minBlurRatio;/,
    );
  });
});

/**
 * W5 (2026-09-24 logic-moves): GET /review/{tab} accepts class_id/source/
 * conf_min/conf_max now — re-enabled unconditionally, not behind the old
 * REVIEW_SERVER_FILTERS_ENABLED flag (deleted). Static source-scan, same
 * pattern as the block above.
 */
describe('review page: class/source/conf filters are sent unconditionally (no server-filters flag)', () => {
  it('never references the deleted REVIEW_SERVER_FILTERS_ENABLED flag', () => {
    expect(src).not.toMatch(/REVIEW_SERVER_FILTERS_ENABLED/);
  });

  it('never references the deleted DEEP_LINK_MAX_ITEMS paging scan', () => {
    expect(src).not.toMatch(/DEEP_LINK_MAX_ITEMS/);
  });

  it('_filter() sends class_id/source/conf_min/conf_max', () => {
    const filterFnStart = src.indexOf('function _filter()');
    const filterFnBody = src.slice(filterFnStart, src.indexOf('\n  }\n', filterFnStart));
    expect(filterFnBody).toMatch(/if \(classFilter != null\) f\.class_id = classFilter;/);
    expect(filterFnBody).toMatch(/if \(sourceFilter\) f\.source = sourceFilter;/);
    expect(filterFnBody).toMatch(/if \(confMin > 0\) f\.conf_min = confMin;/);
    expect(filterFnBody).toMatch(/if \(confMax < 1\) f\.conf_max = confMax;/);
  });

  it("the Source filter bar control is gated on filterVisible('source'), not a hardcoded tab check", () => {
    const idx = src.indexOf('<span class="text-zinc-400">Source</span>');
    expect(idx).toBeGreaterThan(-1);
    const gate = src.slice(src.lastIndexOf('{#if', idx), idx);
    expect(gate).toMatch(/filterVisible\('source'\)/);
    expect(gate).not.toMatch(/PRIMARY_TABS/);
    expect(gate).not.toMatch(/includes\(effectiveTab\)/);
  });
});

/**
 * dq-queues cutover (2026-09-24): the subject/max_rank control's "unset"
 * label used to hardcode "Top 2" — false on any tab whose served
 * `filter_defaults.max_rank` isn't 2 (or is absent). It now reads
 * `servedMaxRankDefault`, sourced from
 * `reviewTabsVocabularyStore.filterDefault(activeTabEndpointId, 'max_rank')`.
 */
describe('review page: subject-scope "unset" label reflects the served max_rank default', () => {
  it('computes servedMaxRankDefault from the served filter_defaults, not a literal 2', () => {
    expect(src).toMatch(
      /const servedMaxRankDefault = \$derived\.by<number \| null>\(\(\) => \{/,
    );
    expect(src).toMatch(
      /reviewTabsVocabularyStore\.filterDefault\(activeTabEndpointId, 'max_rank'\)/,
    );
  });

  it('labels[0] is built from servedMaxRankDefault, with a generic fallback when null', () => {
    const idx = src.indexOf('<SubjectScopeToggle');
    const closeIdx = src.indexOf('/>', idx);
    const block = src.slice(idx, closeIdx);
    expect(block).toMatch(
      /servedMaxRankDefault != null \? `Top \$\{servedMaxRankDefault\}` : 'All ranks'/,
    );
    expect(block).not.toMatch(/^\s*labels=\{\['Top 2'/m);
  });
});
