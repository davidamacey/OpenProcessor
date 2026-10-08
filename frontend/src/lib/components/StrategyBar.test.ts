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
    // Backspace keys into the region bbox canvas while in edit mode. If
    // wiring in <StrategyBar> ever adds a second window.addEventListener
    // call here, this catches it.
    const src = read('../../routes/p/[project]/review/+page.svelte');
    const matches = src.match(/window\.addEventListener\(['"]keydown['"]/g) ?? [];
    expect(matches).toHaveLength(1);
  });

  it('clusters/[id]/+page.svelte registers zero direct window/document keydown listeners', () => {
    // All of this route's shortcuts go through keyboardStore.register(),
    // not a direct addEventListener call in the component itself.
    const src = read('../../routes/p/[project]/clusters/[id]/+page.svelte');
    expect(src).not.toMatch(/(window|document)\.addEventListener\(['"]keydown['"]/);
  });
});

// Phase 4 added the 'diverse' overlay + k stepper (a plain <input
// type="number"> with an `oninput` handler, not a global listener) to
// both StrategyBar.svelte and clusters/[id]/+page.svelte. Re-running the
// exact same assertions post-Phase-4 is the regression guard: the k
// stepper must stay a pointer/keyboard-in-the-input-field-only control,
// never a new global keydown binding (CLAUDE.md's Keyboard shortcuts
// section — reserved keys `g n d z x u a m` stay untouched, and this
// phase adds zero new global keybindings).
// Audit-remediation plan Phase 6 (P1-2/P1-3): StrategyBar's coverage
// gating logic must go through the shared, unit-tested `hasFieldCoverage`
// (strategies.test.ts covers its null-vs-zero cases directly) rather than
// a local reimplementation. This is a static-scan regression guard for the
// exact bug pattern that shipped before this phase -- `?? 0` conflating
// "coverage unknown" with "coverage confirmed zero" -- since this repo has
// no component-mount harness to assert chip/dropdown visibility directly.
describe('coverage gating delegates to the shared hasFieldCoverage (Phase 6)', () => {
  it('imports and calls hasFieldCoverage', () => {
    const src = read('./StrategyBar.svelte');
    expect(src).toMatch(
      /import\s*\{[^}]*hasFieldCoverage[^}]*\}\s*from\s*['"]\$lib\/strategies['"]/,
    );
    expect(src).toMatch(/hasFieldCoverage\(/);
  });

  it('never reintroduces the `field_coverage ?? 0` null-vs-zero bug locally', () => {
    const src = read('./StrategyBar.svelte');
    expect(src).not.toMatch(/field_coverage\s*\?\?\s*0/);
  });
});

describe('Phase 4 (diverse overlay + k stepper) adds no new listeners', () => {
  it('StrategyBar.svelte still calls zero addEventListener with the k stepper present', () => {
    const src = read('./StrategyBar.svelte');
    expect(src).not.toMatch(/addEventListener/);
    // Sanity: the k stepper is actually there, so the above isn't
    // vacuously true against a component that never got the feature.
    expect(src).toMatch(/diverseSelected/);
  });

  it('clusters/[id]/+page.svelte still registers zero direct keydown listeners with diverse wiring present', () => {
    const src = read('../../routes/p/[project]/clusters/[id]/+page.svelte');
    expect(src).not.toMatch(/(window|document)\.addEventListener\(['"]keydown['"]/);
    expect(src).toMatch(/diverseAvailable/);
  });
});
