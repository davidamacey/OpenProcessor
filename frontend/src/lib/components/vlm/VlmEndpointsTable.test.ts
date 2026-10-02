/**
 * The registry table: served order, source / status / locality through the
 * served labels (raw when unlabeled), the served warning chip only for an
 * endpoint that sends crops outside the deployment, key reference and
 * presence (never a key), the projects running it with this one marked,
 * Delete only on stored rows, and the row actions.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import {
  EXTERNAL_WARNING,
  listFixture,
  probeFixture,
  summaryFixture,
} from '$lib/test/fixtures/vlm';
import type { VlmEndpointList } from '$lib/types_vlm';
import VlmEndpointsTable from './VlmEndpointsTable.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

function render(
  over: Record<string, unknown> = {},
  list: VlmEndpointList = listFixture(),
) {
  const props = {
    list,
    currentSlug: 'alpha',
    probes: {},
    probeErrors: {},
    probing: null,
    busy: false,
    onactivate: vi.fn(),
    onprobe: vi.fn(),
    onclone: vi.fn(),
    ondelete: vi.fn(),
    ...over,
  };
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(VlmEndpointsTable, { target, props });
  flushSync();
  return props;
}

const row = (name: string) =>
  target.querySelector<HTMLElement>(
    `[data-testid="vlm-endpoint-row"][data-name="${name}"]`,
  )!;
const within = (el: HTMLElement, id: string) =>
  el.querySelector<HTMLElement>(`[data-testid="${id}"]`);

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
});

describe('VlmEndpointsTable', () => {
  it('lists the endpoints in served order', () => {
    render();
    const names = [...target.querySelectorAll('[data-testid="vlm-endpoint-row"]')].map(
      (r) => r.getAttribute('data-name'),
    );
    expect(names).toEqual(['local_vlm', 'cloud_vlm', 'env_default']);
    expect(target.querySelector('[data-testid="vlm-external-policy"]')?.textContent).toBe(
      'ack',
    );
  });

  it('prints status, locality and source through the served labels, raw when unlabeled', () => {
    const list = listFixture();
    list.endpoints.push(
      summaryFixture({ name: 'odd', status: 'future_status' as never }),
    );
    render({}, list);
    expect(within(row('local_vlm'), 'vlm-endpoint-status')?.textContent).toBe('Ready');
    expect(row('local_vlm').textContent).toContain('Private network');
    expect(row('local_vlm').textContent).toContain('Saved here');
    expect(within(row('cloud_vlm'), 'vlm-endpoint-status')?.textContent).toBe(
      'Not probed yet',
    );
    expect(within(row('odd'), 'vlm-endpoint-status')?.textContent).toBe('future_status');
  });

  it('the served warning chip appears only for an endpoint that sends crops outside', () => {
    render();
    expect(within(row('cloud_vlm'), 'vlm-external-warning')?.textContent).toBe(
      EXTERNAL_WARNING,
    );
    expect(within(row('local_vlm'), 'vlm-external-warning')).toBeNull();
  });

  it('shows the key reference and whether the host has it, never a key', () => {
    render();
    const key = within(row('cloud_vlm'), 'vlm-key-ref')!;
    expect(key.textContent).toContain('CLOUD_VLM_KEY');
    expect(key.textContent).toContain('present on the host');
    expect(within(row('local_vlm'), 'vlm-key-ref')).toBeNull();
    const list = listFixture();
    list.endpoints[1]!.api_key_present = false;
    unmount(instance!);
    target.remove();
    render({}, list);
    expect(within(row('cloud_vlm'), 'vlm-key-ref')?.textContent).toContain(
      'not found on the host',
    );
  });

  it('marks this project among the projects running an endpoint', () => {
    render();
    expect(within(row('local_vlm'), 'vlm-active-in')?.textContent).toContain(
      'alpha (this project)',
    );
    expect(within(row('cloud_vlm'), 'vlm-active-in')?.textContent).toContain('none');
    unmount(instance!);
    target.remove();
    render({ currentSlug: 'beta' });
    expect(within(row('local_vlm'), 'vlm-active-in')?.textContent).not.toContain(
      'this project',
    );
  });

  it('offers Delete on stored rows only; an env row reads read-only', () => {
    render();
    expect(within(row('local_vlm'), 'vlm-delete')).not.toBeNull();
    expect(within(row('cloud_vlm'), 'vlm-delete')).not.toBeNull();
    expect(within(row('env_default'), 'vlm-delete')).toBeNull();
    expect(within(row('env_default'), 'vlm-read-only')).not.toBeNull();
    expect(within(row('local_vlm'), 'vlm-read-only')).toBeNull();
  });

  it('every row offers Activate here, Probe and Clone, each reporting its row', () => {
    const p = render();
    within(row('env_default'), 'vlm-activate')!.click();
    within(row('local_vlm'), 'vlm-probe-btn')!.click();
    within(row('cloud_vlm'), 'vlm-clone')!.click();
    within(row('cloud_vlm'), 'vlm-delete')!.click();
    expect(p.onactivate.mock.calls[0]![0].name).toBe('env_default');
    expect(p.onprobe.mock.calls[0]![0].name).toBe('local_vlm');
    expect(p.onclone.mock.calls[0]![0].name).toBe('cloud_vlm');
    expect(p.ondelete.mock.calls[0]![0].name).toBe('cloud_vlm');
  });

  it('renders a served probe result and a served probe refusal under their rows', () => {
    render({
      probes: { local_vlm: probeFixture() },
      probeErrors: { cloud_vlm: 'A probe is running.' },
    });
    expect(target.querySelector('[data-testid="vlm-probe-ok"]')?.textContent).toBe(
      'probe ok',
    );
    expect(
      target.querySelector('[data-testid="vlm-probe-error"]')?.textContent?.trim(),
    ).toBe('A probe is running.');
  });
});
