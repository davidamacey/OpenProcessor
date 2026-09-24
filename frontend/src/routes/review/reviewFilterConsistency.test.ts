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
 * Disagreements, Plates, Gemma-mismatches chip, Gemma-low-conf chip) — even
 * though the backend (`review.py`) always treats `max_rank` /
 * `min_blur_ratio` as tab-agnostic ("Both apply across tabs"). This is a
 * static source-scan (no @testing-library/svelte in this repo — see
 * clusterMoveRace.test.ts for the established precedent) proving the gate is
 * gone and these controls/filters are unconditional like Conf/Class/HDD
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

  it('renders BlurSlider unconditionally alongside SubjectScopeToggle', () => {
    const scopeIdx = src.indexOf('<SubjectScopeToggle');
    const blurIdx = src.indexOf('<BlurSlider', scopeIdx);
    expect(blurIdx).toBeGreaterThan(scopeIdx);
    // No {/if} closing the gate should sit between the two components.
    const between = src.slice(scopeIdx, blurIdx);
    expect(between).not.toMatch(/\{\/if\}/);
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
