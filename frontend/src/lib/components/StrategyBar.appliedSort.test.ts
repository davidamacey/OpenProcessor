/**
 * Mount-based behavior test for StrategyBar's applied-sort summary
 * (docs/design/test-audit-2026-09-24.md P1-4): the collapsed chip shows
 * "→ <applied>" only when the backend's `sort_applied` differs from what
 * the operator picked, and shows nothing when it matches or is absent.
 * `formatAppliedSort` already has its own pure-function test
 * (strategyBar.svelte.ts); this covers the actual DOM wiring on top of it.
 *
 * Kept as its own file rather than folded into `StrategyBar.test.ts`,
 * which is a source-scan guarding a different, unrelated concern (the
 * pointer-only / zero-global-keydown-listener constraint) that a mount
 * test doesn't replace.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import StrategyBar from './StrategyBar.svelte';
import { createStrategyBar } from '$lib/strategyBar.svelte';
import { FALLBACK_METHODS } from '$lib/strategies';
import { strategiesStore } from '$stores/strategies.svelte';

let target: HTMLDivElement;
let instance: unknown;

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  vi.unstubAllGlobals();
});

function render(props: Record<string, unknown>) {
  // strategiesStore.init() fires on mount; a rejected fetch degrades to
  // FALLBACK_METHODS (documented "never throws" contract) rather than
  // hanging the test on a real network call.
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network in test')));
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(StrategyBar, { target, props } as never);
  flushSync();
  return target;
}

describe('StrategyBar — applied-sort summary', () => {
  it('shows "→ applied" when the backend sort differs from the request', () => {
    const bar = createStrategyBar();
    bar.sort = 'atypicality';
    const el = render({ bar, appliedSort: 'representativeness' });

    expect(el.textContent).toContain('→ representativeness');
  });

  it('shows nothing when the applied sort matches what was requested', () => {
    const bar = createStrategyBar();
    bar.sort = 'atypicality';
    const el = render({ bar, appliedSort: 'atypicality' });

    expect(el.textContent).not.toContain('→');
  });

  // R4 (docs/design/visual-audit-2026-09-24.md): the chip showed raw ids
  // ("→ coco_blind_spots_default") though /methods serves a label for each.
  it('shows the served /methods label for the applied sort, not its raw id', () => {
    strategiesStore.methods = {
      ...FALLBACK_METHODS,
      review_sorts: [
        {
          id: 'primary_low_conf_default',
          axis: 'sort',
          label: 'Largest subject, least confident',
          status: 'stable',
        },
      ],
    } as never;
    const bar = createStrategyBar();
    const el = render({ bar, appliedSort: 'primary_low_conf_default' });

    expect(el.textContent).toContain('→ Largest subject, least confident');
    expect(el.textContent).not.toContain('primary_low_conf_default');
    strategiesStore.methods = FALLBACK_METHODS;
  });

  it('shows nothing when no applied sort was reported', () => {
    const bar = createStrategyBar();
    bar.sort = 'atypicality';
    const el = render({ bar, appliedSort: null });

    expect(el.textContent).not.toContain('→');
  });
});

/**
 * M11 (docs/design/interactive-pass-2026-09-24.md): `sort_fallback_reason`
 * renders inside the collapsed summary chip, next to `sort_applied` —
 * not as a page-level banner the caller has to render separately.
 */
describe('StrategyBar — fallback-reason summary', () => {
  it('shows the fallback reason next to the applied sort when both are set', () => {
    const bar = createStrategyBar();
    bar.sort = 'mistakenness';
    const el = render({
      bar,
      appliedSort: 'atypicality',
      fallbackReason: "'mistakenness' has 0% coverage",
    });

    expect(el.textContent).toContain('→ atypicality');
    expect(el.textContent).toContain("fallback: 'mistakenness' has 0% coverage");
  });

  it('shows nothing when no fallback reason was reported', () => {
    const bar = createStrategyBar();
    const el = render({ bar, appliedSort: null, fallbackReason: null });

    expect(el.textContent).not.toContain('fallback:');
  });
});
