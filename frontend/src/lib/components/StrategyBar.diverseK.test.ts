/**
 * The diverse `k` stepper's upper bound is the served one (`/methods` diverse
 * entry `max_k` / `select_max_k`, passed in as `diverseKMax`); with none
 * served there is no bound and no clamp (no client constant).
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
  if (instance) unmount(instance as never);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
});

function stepper(props: Record<string, unknown>): HTMLInputElement {
  strategiesStore.methods = {
    ...EMPTY_METHODS,
    overlays: [{ id: 'diverse', axis: 'overlay', label: 'Diverse', status: 'stable' }],
  } as never;
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network in test')));
  target = document.createElement('div');
  document.body.appendChild(target);
  const bar = createStrategyBar();
  bar.sort = 'diverse';
  instance = mount(StrategyBar, {
    target,
    props: { bar, offerDiverse: true, ...props },
  } as never);
  flushSync();
  (target.querySelector('button') as HTMLButtonElement).click();
  flushSync();
  return target.querySelector('input[type="number"]') as HTMLInputElement;
}

function type(input: HTMLInputElement, value: string): void {
  input.value = value;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}

describe('StrategyBar diverse k bound', () => {
  it('clamps to the served maximum', () => {
    const input = stepper({ diverseKMax: 10000 });
    expect(input.max).toBe('10000');
    type(input, '20000');
    expect(input.value).toBe('10000');
  });

  it('has no maximum and no clamp when none is served', () => {
    const input = stepper({});
    expect(input.getAttribute('max')).toBeNull();
    type(input, '20000');
    expect(input.value).toBe('20000');
  });
});
