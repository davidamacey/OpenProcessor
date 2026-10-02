import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import CombineNextSteps from './CombineNextSteps.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;
const steps = [
  { action: 'recluster', method: 'POST', path: '/cluster/umap/rebuild', reason: 'r' },
];

function render(ready: boolean, targetStatus: string | null, onpick = vi.fn()) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CombineNextSteps, {
    target,
    props: { steps, ready, targetStatus, onpick },
  });
  flushSync();
  return onpick;
}
const btn = () =>
  target.querySelector<HTMLButtonElement>('[data-testid="combine-next-step-recluster"]')!;
const note = () =>
  target.querySelector('[data-testid="combine-next-step-target-status"]');

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

describe('CombineNextSteps', () => {
  it('disables the button and shows the served status while not ready', () => {
    const onpick = render(false, 'building');
    expect(btn().disabled).toBe(true);
    expect(note()?.textContent).toContain('Building');
    btn().click();
    expect(onpick).not.toHaveBeenCalled();
  });

  it('enables the button, with no status note, once ready', () => {
    const onpick = render(true, 'active');
    expect(btn().disabled).toBe(false);
    expect(note()).toBeNull();
    btn().click();
    expect(onpick).toHaveBeenCalledWith(steps[0]);
  });
});
