/**
 * `/models` VLM rows (W9): one `kind: 'vlm'` row per registered endpoint,
 * the row name is the endpoint name, the served resolved model, the status
 * through the served registry labels (raw when unlabeled), the served
 * "active" chip, no Unload (`unloadable: false`), and a link to Settings →
 * Models only when the registry is served. The old "the VLM is an external
 * service" copy is gone.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ModelsPage from './+page.svelte';
import { vlmAvailability, vlmStatusLabels } from '$lib/vlm/vlmAvailability.svelte';
import { listFixture } from '$lib/test/fixtures/vlm';

const VLM_ROW = {
  role: 'VLM endpoint',
  kind: 'vlm',
  model_type: 'Vision-language model',
  version: null,
  inference_count: null,
  exec_count: null,
  inference_failed: null,
  avg_latency_ms: null,
  last_error: null,
  is_region_protected: false,
  requires_force_to_unload: false,
  unloadable: false,
  optional: false,
  project: null,
  shared: false,
  class_mapping: null,
  owned: false,
  sharing_revision: null,
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;
let registryServed: boolean;

const q = (id: string) => target.querySelector<HTMLElement>(`[data-testid="${id}"]`);

beforeEach(() => {
  vlmAvailability.reset();
  vlmStatusLabels.status = null;
  registryServed = true;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const u = String(url);
      if (u.endsWith('/vlm/endpoints')) {
        return registryServed ? json(listFixture()) : json({ detail: 'Not Found' }, 404);
      }
      if (u.includes('/models/status')) {
        return json({
          models: [
            {
              ...VLM_ROW,
              name: 'local_vlm',
              friendly_name: 'local_vlm',
              status: 'ready',
              model: 'example/vision-7b',
              endpoint: 'http://vlm.internal:8000/v1',
              active: true,
              active_in: ['alpha'],
            },
            {
              ...VLM_ROW,
              name: 'cloud_vlm',
              friendly_name: 'cloud_vlm',
              status: 'unprobed',
              model: null,
              endpoint: 'https://api.example.com/v1',
              active: false,
              active_in: [],
            },
            {
              ...VLM_ROW,
              name: 'odd_vlm',
              friendly_name: 'odd_vlm',
              status: 'future_status',
              model: 'x',
              endpoint: null,
              active: false,
              active_in: [],
            },
          ],
        });
      }
      return json({ detail: 'unrouted' }, 500);
    }),
  );
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
  vi.unstubAllGlobals();
  vlmAvailability.reset();
});

async function render(): Promise<void> {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ModelsPage, { target });
  flushSync();
  await vi.waitFor(() => expect(target.textContent).toContain('local_vlm'));
  await vi.waitFor(() => expect(vlmAvailability.available).not.toBeNull());
  flushSync();
}

const card = (name: string) =>
  [...target.querySelectorAll('li')].find((li) =>
    li.querySelector('p.font-mono')?.textContent?.startsWith(name),
  )!;

describe('/models VLM rows', () => {
  it('lists one row per endpoint, named by the endpoint, with the served model', async () => {
    await render();
    expect(card('local_vlm').querySelector('h2')?.textContent?.trim()).toBe('local_vlm');
    expect(
      card('local_vlm')
        .querySelector('[data-testid="model-vlm-model"]')
        ?.textContent?.trim(),
    ).toBe('example/vision-7b');
    // A null model is "—", never an invented value.
    expect(
      card('cloud_vlm')
        .querySelector('[data-testid="model-vlm-model"]')
        ?.textContent?.trim(),
    ).toBe('—');
  });

  it('names the status through the served labels, raw when unlabeled', async () => {
    await render();
    const pill = (n: string) =>
      card(n).querySelector('[data-testid="model-status-pill"]')?.textContent?.trim();
    expect(pill('local_vlm')).toBe('Ready');
    expect(pill('cloud_vlm')).toBe('Not probed yet');
    expect(pill('odd_vlm')).toBe('future_status');
  });

  it('shows the active chip only for the served active endpoint, and no Unload', async () => {
    await render();
    expect(
      card('local_vlm').querySelector('[data-testid="model-vlm-active"]'),
    ).not.toBeNull();
    expect(
      card('cloud_vlm').querySelector('[data-testid="model-vlm-active"]'),
    ).toBeNull();
    expect(target.textContent).not.toContain('Unload');
  });

  it('links to Settings → Models when the registry is served', async () => {
    await render();
    expect(
      card('local_vlm')
        .querySelector('[data-testid="model-vlm-manage"]')
        ?.getAttribute('href'),
    ).toMatch(/\/settings\/models$/);
  });

  it('shows no link (absent, not disabled) when the registry is not served', async () => {
    registryServed = false;
    await render();
    expect(q('model-vlm-manage')).toBeNull();
  });

  it('no longer calls the VLM an external service', async () => {
    await render();
    expect(target.textContent).not.toContain('the VLM is an external service');
    expect(target.textContent).toContain('Settings → Models');
  });
});
