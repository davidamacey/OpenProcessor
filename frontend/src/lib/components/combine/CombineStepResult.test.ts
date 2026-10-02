import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import CombineStepResult from './CombineStepResult.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

function render(action: string, result: unknown) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CombineStepResult, { target, props: { action, result } });
  flushSync();
}
const q = (id: string) => target.querySelector(`[data-testid="${id}"]`);

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

describe('CombineStepResult', () => {
  it('renders the live no_residuals body: label, status chip, scalar rows', () => {
    render('recluster', { status: 'no_residuals', n_residuals: 0, refit: false });
    expect(q('combine-step-result-action')?.textContent).toBe('Recluster');
    expect(q('combine-step-result-status')?.textContent).toBe('No residuals');
    expect(target.querySelector('[data-result-key="n_residuals"]')?.textContent).toBe(
      '0',
    );
    expect(target.querySelector('[data-result-key="refit"]')?.textContent).toBe('false');
    expect(target.querySelector('[data-result-key="status"]')).toBeNull();
  });

  it('collapses nested and long values, and tolerates an empty body', () => {
    render('recluster', { n: 1, nested: { a: 1 }, long: 'x'.repeat(100) });
    expect(target.querySelector('[data-result-key="nested"]')).toBeNull();
    expect(target.querySelector('[data-result-key="long"]')).toBeNull();
    expect(target.querySelectorAll('details')).toHaveLength(2);
    expect(q('combine-step-result-status')?.textContent).toBe('Done');
  });
});
