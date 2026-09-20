/**
 * `<AssistScopeBar>` is a pointer-only, collapse/expand-on-click control,
 * same convention as `StrategyBar.svelte` — it must never register a
 * window/document keydown listener (CLAUDE.md's Keyboard shortcuts
 * section: two window keydown listeners already fire on every keypress
 * app-wide, and a third would be a real collision).
 *
 * This repo has no `@testing-library/svelte`, so — same as
 * `StrategyBar.test.ts` — this is a static source scan rather than a
 * mounted-component assertion.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));

function read(rel: string): string {
  return readFileSync(path.resolve(here, rel), 'utf-8');
}

describe('AssistScopeBar.svelte', () => {
  const src = read('./AssistScopeBar.svelte');

  it('never calls addEventListener — pointer-only, zero global listeners', () => {
    expect(src).not.toMatch(/addEventListener/);
  });

  it('imports selectableAxisEntries, isDetectionProfileAvailable, isPromptPackAvailable from $lib/strategies', () => {
    expect(src).toMatch(
      /import\s*\{[^}]*selectableAxisEntries[^}]*\}\s*from\s*['"]\$lib\/strategies['"]/s,
    );
    expect(src).toMatch(/isDetectionProfileAvailable/);
    expect(src).toMatch(/isPromptPackAvailable/);
  });

  it('imports searchClasses from $lib/classPicker', () => {
    expect(src).toMatch(
      /import\s*\{[^}]*searchClasses[^}]*\}\s*from\s*['"]\$lib\/classPicker['"]/,
    );
  });

  // The whole reason selectableAxisEntries exists (strategies.ts §2.4) is
  // that a second, divergent copy of the status predicate is the
  // hasFieldCoverage bug class. This component must go through the shared
  // helper, never re-derive the filter inline.
  it('contains no inline status filter', () => {
    expect(src).not.toMatch(/status === 'stable'/);
  });

  it('gates the detector <select> behind detectionProfileAvailable', () => {
    expect(src).toMatch(/\{#if detectionProfileAvailable\}/);
  });

  it('gates the prompts <select> behind promptPackAvailable', () => {
    expect(src).toMatch(/\{#if promptPackAvailable\}/);
  });

  it('reads strategiesStore.methods.detection_profiles and .prompt_packs, and calls strategiesStore.init()', () => {
    expect(src).toMatch(/strategiesStore\.methods\.detection_profiles/);
    expect(src).toMatch(/strategiesStore\.methods\.prompt_packs/);
    expect(src).toMatch(/strategiesStore\.init\(\)/);
  });
});
