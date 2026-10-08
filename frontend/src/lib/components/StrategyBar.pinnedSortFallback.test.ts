/**
 * Mount-based behavior test for StrategyBar's pinned-default-coverage
 * chip (visual-audit S1's last bullet, `docs/design/
 * visual-audit-2026-09-24.md`, corrected status: the "deferred, BACKEND"
 * note was wrong — `GET {API_PREFIX}/methods` already serves real
 * `field_coverage` on every `sort` entry, and `hasFieldCoverage`
 * already gates it; this just wires that existing signal into the
 * summary chip). `formatPinnedSortFallback` has its own pure-function
 * coverage in `strategyBar.svelte.test.ts`; this covers the actual DOM
 * wiring — including that it MERGES with, rather than duplicates, the
 * plain "→ applied" mismatch chip `StrategyBar.appliedSort.test.ts`
 * covers.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import StrategyBar from './StrategyBar.svelte';
import { createStrategyBar } from '$lib/strategyBar.svelte';
import { EMPTY_METHODS } from '$lib/strategies';
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
  strategiesStore.methods = EMPTY_METHODS;
});

function render(props: Record<string, unknown>) {
  // strategiesStore.init() fires on mount; a rejected fetch leaves
  // EMPTY_METHODS (init() never throws) rather than
  // hanging the test on a real network call.
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network in test')));
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(StrategyBar, { target, props } as never);
  flushSync();
  return target;
}

function withZeroCoverageUncertainty(): void {
  strategiesStore.methods = {
    ...EMPTY_METHODS,
    review_sorts: [
      {
        id: 'uncertainty_entropy',
        label: 'Uncertainty (probe entropy)',
        status: 'stable',
        requires_field: 'probe_entropy',
        field_coverage: 0,
        field_coverage_total: 7961,
      },
      {
        id: 'atypicality',
        label: 'Atypicality (outlier-first)',
        status: 'stable',
        requires_field: 'atypicality_score',
        field_coverage: 450,
        field_coverage_total: 7961,
      },
    ],
  } as never;
}

describe('StrategyBar — pinned-default-coverage chip', () => {
  it('shows the chip with the served sort_applied when the pinned default has 0 coverage', () => {
    withZeroCoverageUncertainty();
    const bar = createStrategyBar();
    // bar.sort stays on the sentinel ('default') — no operator override.
    const el = render({
      bar,
      appliedSort: 'atypicality',
      pinnedSortId: 'uncertainty_entropy',
    });

    const chip = el.querySelector('[data-testid="pinned-sort-fallback-chip"]');
    expect(chip).not.toBeNull();
    expect(chip!.textContent).toContain('Uncertainty (probe entropy)');
    expect(chip!.textContent).toContain('Atypicality (outlier-first)');
    // Merged, not doubled: the plain "→ applied" mismatch text must not
    // also render alongside the pinned-specific chip.
    expect(el.textContent).not.toContain('→ Atypicality (outlier-first)');
  });

  it('shows no chip when the pinned default has real coverage', () => {
    strategiesStore.methods = {
      ...EMPTY_METHODS,
      review_sorts: [
        {
          id: 'representativeness',
          label: 'Representativeness',
          status: 'stable',
          requires_field: 'representativeness_score',
          field_coverage: 12000,
          field_coverage_total: 12000,
        },
      ],
    } as never;
    const bar = createStrategyBar();
    const el = render({
      bar,
      appliedSort: 'representativeness',
      pinnedSortId: 'representativeness',
    });

    expect(el.querySelector('[data-testid="pinned-sort-fallback-chip"]')).toBeNull();
  });

  it('shows no chip once the operator has picked their own sort override', () => {
    withZeroCoverageUncertainty();
    const bar = createStrategyBar();
    bar.sort = 'atypicality'; // operator override — no longer "unset"
    const el = render({
      bar,
      appliedSort: 'atypicality',
      pinnedSortId: 'uncertainty_entropy',
    });

    expect(el.querySelector('[data-testid="pinned-sort-fallback-chip"]')).toBeNull();
  });

  it('shows no chip when no pinned default was resolved for this tab', () => {
    withZeroCoverageUncertainty();
    const bar = createStrategyBar();
    const el = render({ bar, appliedSort: 'atypicality', pinnedSortId: null });

    expect(el.querySelector('[data-testid="pinned-sort-fallback-chip"]')).toBeNull();
  });
});
