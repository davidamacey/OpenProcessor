/**
 * The per-run VLM picker: absent (nothing rendered, no request of its own)
 * without a usable `vlm` axis; "Project default" first, then every served
 * entry that is not disabled with its served status label; the served
 * warning banner and the "I understand" checkbox only for an entry whose
 * served `per_run_ack_required` is true; a new pick drops an earlier
 * acknowledgement; the checkbox reports the acknowledgement.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { EMPTY_METHODS, parseMethodsResponse } from '$lib/strategies';
import { strategiesStore } from '$stores/strategies.svelte';
import VlmRunPicker from './VlmRunPicker.svelte';

const WIRE = {
  strategies: [
    {
      id: 'local_vlm',
      axis: 'vlm',
      label: 'Local VLM',
      status: 'stable',
      endpoint_status_label: 'Ready',
      per_run_ack_required: false,
    },
    {
      id: 'cloud_vlm',
      axis: 'vlm',
      label: 'Cloud VLM',
      status: 'experimental',
      endpoint_status_label: 'Not probed yet',
      warning: 'Crops leave the deployment.',
      per_run_ack_required: true,
    },
    { id: 'retired', axis: 'vlm', label: 'Retired', status: 'disabled' },
  ],
};

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

function render(props: Record<string, unknown>) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(VlmRunPicker, { target, props } as never);
  flushSync();
}

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

describe('VlmRunPicker', () => {
  it('renders nothing when /methods serves no vlm axis (absent, not disabled)', () => {
    strategiesStore.methods = EMPTY_METHODS;
    render({ vlm: null, acknowledgeExternal: false, onchange: vi.fn() });
    expect(q('vlm-run-picker')).toBeNull();
    expect(target.querySelector('select')).toBeNull();
  });

  it('renders nothing when every entry is disabled', () => {
    strategiesStore.methods = parseMethodsResponse({
      strategies: [{ id: 'retired', axis: 'vlm', label: 'Retired', status: 'disabled' }],
    });
    render({ vlm: null, acknowledgeExternal: false, onchange: vi.fn() });
    expect(q('vlm-run-picker')).toBeNull();
  });

  it('lists Project default first, then the served entries with their status label', () => {
    strategiesStore.methods = parseMethodsResponse(WIRE);
    render({ vlm: null, acknowledgeExternal: false, onchange: vi.fn() });
    const opts = [...target.querySelectorAll('option')].map((o) => [
      o.value,
      o.textContent?.trim(),
    ]);
    expect(opts).toEqual([
      ['', 'Project default'],
      ['local_vlm', 'Local VLM · Ready'],
      ['cloud_vlm', 'Cloud VLM · Not probed yet'],
    ]);
    expect(q('vlm-run-ack')).toBeNull();
  });

  it('shows the served warning and the checkbox only when per_run_ack_required is true', () => {
    strategiesStore.methods = parseMethodsResponse(WIRE);
    render({ vlm: 'local_vlm', acknowledgeExternal: false, onchange: vi.fn() });
    expect(q('vlm-run-ack')).toBeNull();
    unmount(instance!);
    target.remove();
    render({ vlm: 'cloud_vlm', acknowledgeExternal: false, onchange: vi.fn() });
    expect(q('vlm-run-ack')?.textContent).toContain('Crops leave the deployment.');
    expect(q('vlm-run-ack-checkbox')).not.toBeNull();
  });

  it('a pick reports the id and starts without an acknowledgement; Project default reports null', () => {
    strategiesStore.methods = parseMethodsResponse(WIRE);
    const onchange = vi.fn();
    render({ vlm: 'cloud_vlm', acknowledgeExternal: true, onchange });
    const select = q('vlm-run-select') as HTMLSelectElement;
    select.value = 'local_vlm';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onchange).toHaveBeenLastCalledWith({
      vlm: 'local_vlm',
      acknowledgeExternal: false,
    });
    select.value = '';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onchange).toHaveBeenLastCalledWith({ vlm: null, acknowledgeExternal: false });
  });

  it('the checkbox reports the acknowledgement and keeps the pick', () => {
    strategiesStore.methods = parseMethodsResponse(WIRE);
    const onchange = vi.fn();
    render({ vlm: 'cloud_vlm', acknowledgeExternal: false, onchange });
    const box = q('vlm-run-ack-checkbox') as HTMLInputElement;
    box.checked = true;
    box.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onchange).toHaveBeenLastCalledWith({
      vlm: 'cloud_vlm',
      acknowledgeExternal: true,
    });
  });
});
