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

  it('shows nothing when no applied sort was reported', () => {
    const bar = createStrategyBar();
    bar.sort = 'atypicality';
    const el = render({ bar, appliedSort: null });

    expect(el.textContent).not.toContain('→');
  });
});
