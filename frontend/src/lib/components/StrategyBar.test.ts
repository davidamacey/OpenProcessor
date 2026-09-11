/**
 * `<StrategyBar>` is a pointer-only, collapse/expand-on-click control
 * (curation-strategy plan §5.1/§5.4) — it must never register a
 * window/document keydown listener, matching the hard constraint
 * CLAUDE.md's Keyboard shortcuts section places on every new control in
 * this phase.
 *
 * This repo has no `@testing-library/svelte` (and adding one is a new
 * dev dependency this phase doesn't need), so there's no component-mount
 * harness available the way `strategies.svelte.test.ts` spies on
 * `window.addEventListener` around a running store. Instead this test
 * does the same job the way a reviewer would: a static source scan
 * asserting the component's `<script>` never calls `addEventListener`
 * at all, plus a scan of the two routes it's wired into confirming the
 * wiring didn't add a new global keydown listener there either.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));

function read(rel: string): string {
  return readFileSync(path.resolve(here, rel), 'utf-8');
}

describe('StrategyBar.svelte', () => {
  it('never calls addEventListener — pointer-only, zero global listeners', () => {
    const src = read('./StrategyBar.svelte');
    expect(src).not.toMatch(/addEventListener/);
  });
});

describe('routes wired to <StrategyBar> keep their keydown listener count unchanged', () => {
  it('review/+page.svelte still has exactly the one pre-existing window keydown forward', () => {
    // Pre-existing (not introduced by this phase): forwards arrow/[/]/
    // Backspace keys into the plate bbox canvas while in edit mode. If
    // wiring in <StrategyBar> ever adds a second window.addEventListener
    // call here, this catches it.
    const src = read('../../routes/review/+page.svelte');
    const matches = src.match(/window\.addEventListener\(['"]keydown['"]/g) ?? [];
    expect(matches).toHaveLength(1);
  });

  it('clusters/[id]/+page.svelte registers zero direct window/document keydown listeners', () => {
    // All of this route's shortcuts go through keyboardStore.register(),
    // not a direct addEventListener call in the component itself.
    const src = read('../../routes/clusters/[id]/+page.svelte');
    expect(src).not.toMatch(/(window|document)\.addEventListener\(['"]keydown['"]/);
  });
});
