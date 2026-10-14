import { describe, expect, it } from 'vitest';
import { reviewPageSource, reviewSource } from '$lib/testing/reviewPageSources';

/**
 * Regression test for the "tabs are inconsistent" bug (2026-09): the
 * SubjectScopeToggle (rank-scope) and BlurSlider (clarity) controls used to
 * be gated behind `{#if PRIMARY_TABS.includes(effectiveTab)}`, so they
 * appeared only on the Primary·Low-Conf preset chip and the Classifier Blind Spots
 * tab and silently vanished everywhere else (All, Uncertainty, Model
 * Disagreements, Regions, VLM-mismatches chip, VLM-low-conf chip) — even
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
// The whole page (route, controllers, components) for "nowhere" scans; the
// filter bar markup, the filter controller and the tabs controller for the
// code each of them now holds.
const pageSrc = reviewPageSource();
const barSrc = reviewSource('lib/components/review/ReviewFilterBar.svelte');
const filterSrc = reviewSource('lib/review/reviewFilterController.svelte.ts');
const tabsSrc = reviewSource('lib/review/reviewTabsController.svelte.ts');

describe('review page: rank-scope + blur controls are tab-agnostic (no PRIMARY_TABS gate)', () => {
  it('never references PRIMARY_TABS anywhere in the page', () => {
    expect(pageSrc).not.toMatch(/PRIMARY_TABS/);
  });

  it('does not import PRIMARY_TABS from reviewTabs', () => {
    expect(pageSrc).not.toMatch(
      /import\s*\{[^}]*PRIMARY_TABS[^}]*\}\s*from\s*['"]\$lib\/reviewTabs['"]/,
    );
  });

  it('renders SubjectScopeToggle with no #if wrapper gating it to specific tabs', () => {
    const idx = barSrc.indexOf('<SubjectScopeToggle');
    expect(idx).toBeGreaterThan(-1);
    // Walk backward from the component tag to the nearest control-flow
    // block opener; it must not be an {#if ...includes(effectiveTab)...}
    // guard (or any {#if} at all immediately wrapping just these controls).
    const preceding = barSrc.slice(0, idx);
    const lastIfMatch = [...preceding.matchAll(/\{#if [^}]*\}/g)].pop();
    if (lastIfMatch) {
      expect(lastIfMatch[0]).not.toMatch(/includes\(effectiveTab\)/);
      expect(lastIfMatch[0]).not.toMatch(/PRIMARY_TABS/);
    }
  });

  it('gates SubjectScopeToggle/BlurSlider on filterVisible(...), never on PRIMARY_TABS/effectiveTab', () => {
    const scopeIdx = barSrc.indexOf('<SubjectScopeToggle');
    const blurIdx = barSrc.indexOf('<BlurSlider', scopeIdx);
    expect(blurIdx).toBeGreaterThan(scopeIdx);
    const between = barSrc.slice(scopeIdx, blurIdx);
    // A gate may sit between them now (each control has its own served-
    // filter check), but it must be the filterVisible one, not a
    // hardcoded tab-id list.
    expect(between).toMatch(/<SubjectScopeToggle/);
    expect(between).not.toMatch(/PRIMARY_TABS/);
    expect(between).not.toMatch(/includes\(effectiveTab\)/);

    const scopeBlock = barSrc.slice(barSrc.lastIndexOf('{#if', scopeIdx), scopeIdx);
    expect(scopeBlock).toMatch(/filterVisible\('max_rank'\)/);
    const blurBlock = barSrc.slice(barSrc.lastIndexOf('{#if', blurIdx), blurIdx);
    expect(blurBlock).toMatch(/filterVisible\('min_blur_ratio'\)/);
  });

  it('applies max_rank whenever subjectScope is set, regardless of tab (no PRIMARY_TABS.includes guard in _filter)', () => {
    const filterFnStart = filterSrc.indexOf('function _filter()');
    const filterFnBody = filterSrc.slice(
      filterFnStart,
      filterSrc.indexOf('\n  }\n', filterFnStart),
    );
    expect(filterFnBody).toMatch(/if \(subjectScope !== 0\) f\.max_rank = subjectScope;/);
    expect(filterFnBody).not.toMatch(/PRIMARY_TABS/);
    expect(filterFnBody).not.toMatch(/includes\(effectiveTab\)/);
  });

  it('applies min_blur_ratio whenever minBlurRatio is set, regardless of tab', () => {
    const filterFnStart = filterSrc.indexOf('function _filter()');
    const filterFnBody = filterSrc.slice(
      filterFnStart,
      filterSrc.indexOf('\n  }\n', filterFnStart),
    );
    expect(filterFnBody).toMatch(
      /if \(minBlurRatio != null\) f\.min_blur_ratio = minBlurRatio;/,
    );
  });
});

/**
 * GET /review/{tab} accepts class_id/source/
 * conf_min/conf_max now — re-enabled unconditionally, not behind the old
 * REVIEW_SERVER_FILTERS_ENABLED flag (deleted). Static source-scan, same
 * pattern as the block above.
 */
describe('review page: class/source/conf filters are sent unconditionally (no server-filters flag)', () => {
  it('never references the deleted REVIEW_SERVER_FILTERS_ENABLED flag', () => {
    expect(pageSrc).not.toMatch(/REVIEW_SERVER_FILTERS_ENABLED/);
  });

  it('never references the deleted DEEP_LINK_MAX_ITEMS paging scan', () => {
    expect(pageSrc).not.toMatch(/DEEP_LINK_MAX_ITEMS/);
  });

  it('_filter() sends the shared item filter (class by name) and source/conf_min/conf_max', () => {
    const filterFnStart = filterSrc.indexOf('function _filter()');
    const filterFnBody = filterSrc.slice(
      filterFnStart,
      filterSrc.indexOf('\n  }\n', filterFnStart),
    );
    expect(filterFnBody).toMatch(/Object\.assign\(f, itemFilterQuery\);/);
    expect(filterFnBody).not.toMatch(/class_id/);
    expect(filterFnBody).toMatch(/if \(sourceFilter\) f\.source = sourceFilter;/);
    expect(filterFnBody).toMatch(/if \(confMin > 0\) f\.conf_min = confMin;/);
    expect(filterFnBody).toMatch(/if \(confMax < 1\) f\.conf_max = confMax;/);
  });

  it("the Source filter bar control is gated on filterVisible('source'), not a hardcoded tab check", () => {
    const idx = barSrc.indexOf('<span class="text-zinc-400">Source</span>');
    expect(idx).toBeGreaterThan(-1);
    const gate = barSrc.slice(barSrc.lastIndexOf('{#if', idx), idx);
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
    expect(tabsSrc).toMatch(
      /const servedMaxRankDefault = \$derived\.by<number \| null>\(\(\) => \{/,
    );
    expect(tabsSrc).toMatch(
      /reviewTabsVocabularyStore\.filterDefault\(activeTabEndpointId, 'max_rank'\)/,
    );
  });

  it('labels[0] is built from servedMaxRankDefault, with a generic fallback when null', () => {
    const idx = barSrc.indexOf('<SubjectScopeToggle');
    const closeIdx = barSrc.indexOf('/>', idx);
    const block = barSrc.slice(idx, closeIdx);
    expect(block).toMatch(
      /servedMaxRankDefault != null \? `Top \$\{servedMaxRankDefault\}` : 'All ranks'/,
    );
    expect(block).not.toMatch(/^\s*labels=\{\['Top 2'/m);
  });
});
