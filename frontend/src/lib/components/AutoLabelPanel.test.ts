/**
 * Static source-scan regression guard for the scoped-VLM-assist wiring in
 * `AutoLabelPanel.svelte` (docs/design/vlm-scoped-labeling-assist-plan-2026-09-20.md
 * §4.3). Same convention as `StrategyBar.test.ts`/`AssistScopeBar.test.ts`
 * — this repo has no `@testing-library/svelte`, so a mounted-component
 * assertion isn't available; a source scan is the established substitute.
 *
 * The core property under test: the six pre-existing `startAutoLabel(...)`
 * params must survive verbatim (this is additive, not a rewrite), the
 * panel must never hand-build `class_id:` itself (the only producer is
 * `scope.toStartParams()`), and it must not add a second
 * `classesStore.acquire()` (the layout already owns that ref-count).
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));

function read(rel: string): string {
  return readFileSync(path.resolve(here, rel), 'utf-8');
}

describe('AutoLabelPanel.svelte', () => {
  const src = read('./AutoLabelPanel.svelte');

  it('spreads ...scope.toStartParams() into the startAutoLabel call', () => {
    expect(src).toMatch(/\.\.\.scope\.toStartParams\(\)/);
  });

  it('the six pre-existing start params survive verbatim', () => {
    for (const key of [
      'train_clusters:',
      'vlm_concurrency:',
      'max_vlm_crops:',
      'recluster_unvalidated:',
      'gate_max_rank:',
      'gate_min_blur_ratio:',
      'n_clusters:',
    ]) {
      expect(src).toContain(key);
    }
  });

  it('never hand-builds class_id — the only producer is toStartParams()', () => {
    expect(src).not.toMatch(/class_id:/);
  });

  it('renders <AssistScopeBar inside an {#if scopeAvailable} guard', () => {
    expect(src).toMatch(/\{#if scopeAvailable\}\s*<AssistScopeBar/);
  });

  it("the unscoped button label is still the exact string 'Recluster now'", () => {
    expect(src).toContain('Recluster now');
  });

  it('calls isScopedAssistAvailable and strategiesStore.init()', () => {
    expect(src).toMatch(/isScopedAssistAvailable/);
    expect(src).toMatch(/strategiesStore\.init\(\)/);
  });

  it('does not call classesStore.acquire() — the layout already owns that ref-count', () => {
    expect(src).not.toMatch(/classesStore\.acquire\(\)/);
  });
});
