/**
 * The assist bar's VLM control (W9): expanded, it shows the picker only
 * when `/methods` serves a usable `vlm` axis; a pick lands in the scope
 * (`vlm`, and `acknowledge_external` only after the checkbox) and the
 * collapsed chip names it; a stale pick is dropped when the axis goes
 * away; reset clears both.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { createAssistScope, type AssistScope } from '$lib/assistScope.svelte';
import { EMPTY_METHODS, parseMethodsResponse } from '$lib/strategies';
import { strategiesStore } from '$stores/strategies.svelte';
import AssistScopeBar from './AssistScopeBar.svelte';

const WIRE = {
  strategies: [
    { id: 'local_vlm', axis: 'vlm', label: 'Local VLM', status: 'stable' },
    {
      id: 'cloud_vlm',
      axis: 'vlm',
      label: 'Cloud VLM',
      status: 'stable',
      warning: 'Crops leave the deployment.',
      per_run_ack_required: true,
    },
  ],
};

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;
let scope: AssistScope;

function render() {
  scope = createAssistScope();
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(AssistScopeBar, { target, props: { scope, classes: [] } });
  flushSync();
}

const expand = () => {
  target.querySelector<HTMLButtonElement>('button')!.click();
  flushSync();
};
const q = (id: string) => target.querySelector<HTMLElement>(`[data-testid="${id}"]`);

beforeEach(() => {
  strategiesStore.loaded = true;
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
  strategiesStore.methods = EMPTY_METHODS;
  strategiesStore.loaded = false;
});

describe('AssistScopeBar VLM control', () => {
  it('has no picker without a vlm axis', () => {
    strategiesStore.methods = EMPTY_METHODS;
    render();
    expand();
    expect(q('vlm-run-picker')).toBeNull();
  });

  it('a pick lands in the scope and the collapsed chip names it', () => {
    strategiesStore.methods = parseMethodsResponse(WIRE);
    render();
    expand();
    const select = q('vlm-run-select') as HTMLSelectElement;
    select.value = 'local_vlm';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(scope.vlm).toBe('local_vlm');
    expect(scope.toStartParams()).toEqual({ vlm: 'local_vlm' });
    // Collapse by the × button: the chip summary names the endpoint.
    [...target.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === '×')!
      .click();
    flushSync();
    expect(target.textContent).toContain('whole dataset · local_vlm');
  });

  it('the acknowledgement is part of the scope only once the box is checked', () => {
    strategiesStore.methods = parseMethodsResponse(WIRE);
    render();
    expand();
    const select = q('vlm-run-select') as HTMLSelectElement;
    select.value = 'cloud_vlm';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(scope.toStartParams()).toEqual({ vlm: 'cloud_vlm' });
    const box = q('vlm-run-ack-checkbox') as HTMLInputElement;
    box.checked = true;
    box.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(scope.toStartParams()).toEqual({
      vlm: 'cloud_vlm',
      acknowledge_external: true,
    });
    [...target.querySelectorAll('button')]
      .find((b) => b.textContent?.trim() === 'reset')!
      .click();
    flushSync();
    expect(scope.toStartParams()).toEqual({});
  });

  it('drops a stale pick when the axis stops being served', () => {
    strategiesStore.methods = parseMethodsResponse(WIRE);
    render();
    scope.vlm = 'cloud_vlm';
    scope.acknowledgeExternal = true;
    strategiesStore.methods = EMPTY_METHODS;
    flushSync();
    expect(scope.vlm).toBeNull();
    expect(scope.acknowledgeExternal).toBe(false);
  });
});
