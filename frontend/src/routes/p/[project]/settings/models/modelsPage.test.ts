/**
 * `/settings/models` mounted against stubbed W9 routes: absent (one line,
 * no other `/vlm/*` request) when the registry 404s; otherwise the active
 * panel with the served health block, the endpoints, the local model and
 * the model choices; "Activate here" for an external endpoint shows the
 * served warning and sends `acknowledge_external` only after the checkbox,
 * and a stored endpoint's Delete shows the served `in_use` projects.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ModelsSettingsPage from './+page.svelte';
import { vlmAvailability } from '$lib/vlm/vlmAvailability.svelte';
import { healthStore } from '$stores/health.svelte';
import {
  activeFixture,
  catalogFixture,
  listFixture,
  EXTERNAL_WARNING,
} from '$lib/test/fixtures/vlm';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

class FakeEventSource {
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  addEventListener(): void {}
  close(): void {}
}

let requests: { method: string; url: string; body: unknown }[];
let registryServed: boolean;
let activateResponses: Response[];
let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

const q = (id: string) => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);

beforeEach(() => {
  requests = [];
  registryServed = true;
  activateResponses = [];
  vlmAvailability.reset();
  vi.stubGlobal('EventSource', FakeEventSource);
  healthStore.scopedHealth = {
    status: 'ok',
    project: 'p',
    region_profile: null,
    vlm: {
      reachable: true,
      model: 'example/vision-7b',
      detail: 'serving',
      last_error: null,
    },
  };
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const u = String(url);
      const method = init.method ?? 'GET';
      requests.push({
        method,
        url: u,
        body: init.body ? JSON.parse(String(init.body)) : undefined,
      });
      if (u.endsWith('/vlm/endpoints') && method === 'GET') {
        return registryServed ? json(listFixture()) : json({ detail: 'Not Found' }, 404);
      }
      if (u.endsWith('/vlm/catalog')) return json(catalogFixture());
      if (u.endsWith('/vlm/endpoints/active')) return json(activeFixture());
      if (u.endsWith('/activate'))
        return activateResponses.shift() ?? json(activeFixture());
      if (u.includes('/config/vocabulary')) {
        return json({
          detectors: [],
          segmenters: [],
          vlm: { active: { name: null }, endpoints: [] },
          ocr: { available: false, pipeline_models: [], det_models: [], rec_models: [] },
          text_reader_modes: [],
          registry_classes: [],
          model_choices: [
            {
              role: 'vlm',
              label: 'VLM endpoint',
              scope: 'per_run',
              current: 'local_vlm',
              choices: [],
              settable: true,
            },
          ],
          labels: { scope: { per_run: 'Chosen per run' } },
        });
      }
      if (method === 'DELETE') {
        return json(
          {
            detail: {
              error: 'in_use',
              message: 'local_vlm is active in other projects.',
              projects: ['alpha', 'beta'],
            },
          },
          409,
        );
      }
      return json({ detail: 'Not Found' }, 404);
    }),
  );
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
  document.querySelectorAll('[role="dialog"]').forEach((d) => d.remove());
  vi.unstubAllGlobals();
  vlmAvailability.reset();
  healthStore.scopedHealth = null;
});

async function render(): Promise<void> {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ModelsSettingsPage, { target });
  flushSync();
  await vi.waitFor(() => expect(vlmAvailability.available).not.toBeNull());
  flushSync();
}

const dialogButton = (label: string) =>
  [...document.querySelectorAll('[role="dialog"] button')].find(
    (b) => b.textContent?.trim() === label,
  )!;

describe('/settings/models', () => {
  it('is absent when the registry 404s: one line, and no other /vlm request', async () => {
    registryServed = false;
    await render();
    expect(q('vlm-unavailable')).not.toBeNull();
    expect(requests.filter((r) => r.url.includes('/vlm/'))).toHaveLength(1);
    expect(q('vlm-endpoints')).toBeNull();
  });

  it('renders the active panel, the served health, the endpoints, the local model and the choices', async () => {
    await render();
    await vi.waitFor(() => expect(q('local-vlm')).not.toBeNull());
    await vi.waitFor(() => expect(q('model-choices')).not.toBeNull());
    expect(q('active-ref')?.textContent).toBe('local_vlm r3');
    expect(q('vlm-health')?.textContent).toContain('reachable');
    expect(q('vlm-health')?.textContent).toContain('example/vision-7b');
    expect(document.querySelectorAll('[data-testid="vlm-endpoint-row"]')).toHaveLength(3);
    expect(q('model-choice-row')?.textContent).toContain('Chosen per run');
  });

  it('Activate here on an external endpoint shows the warning and sends the ack only when checked', async () => {
    await render();
    await vi.waitFor(() => expect(q('vlm-endpoints')).not.toBeNull());
    document
      .querySelector<HTMLElement>(
        '[data-testid="vlm-endpoint-row"][data-name="cloud_vlm"] [data-testid="vlm-activate"]',
      )!
      .click();
    flushSync();
    expect(q('activate-external-warning')?.textContent?.trim()).toBe(EXTERNAL_WARNING);
    dialogButton('Activate').dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.waitFor(() =>
      expect(requests.some((r) => r.url.endsWith('/activate'))).toBe(true),
    );
    const first = requests.find((r) => r.url.endsWith('/activate'))!;
    expect(first.url).toContain('/vlm/endpoints/cloud_vlm/activate');
    expect(first.body).toMatchObject({ revision: 1, force: false });
    expect(first.body).not.toHaveProperty('acknowledge_external');
  });

  it('with the box checked, the activation carries acknowledge_external: true', async () => {
    await render();
    await vi.waitFor(() => expect(q('vlm-endpoints')).not.toBeNull());
    document
      .querySelector<HTMLElement>(
        '[data-testid="vlm-endpoint-row"][data-name="cloud_vlm"] [data-testid="vlm-activate"]',
      )!
      .click();
    flushSync();
    q('activate-ack')!.click();
    flushSync();
    dialogButton('Activate').dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.waitFor(() =>
      expect(requests.some((r) => r.url.endsWith('/activate'))).toBe(true),
    );
    expect(requests.find((r) => r.url.endsWith('/activate'))!.body).toMatchObject({
      acknowledge_external: true,
    });
  });

  it('Delete shows the served in_use message and project slugs', async () => {
    await render();
    await vi.waitFor(() => expect(q('vlm-endpoints')).not.toBeNull());
    document
      .querySelector<HTMLElement>(
        '[data-testid="vlm-endpoint-row"][data-name="local_vlm"] [data-testid="vlm-delete"]',
      )!
      .click();
    flushSync();
    dialogButton('Delete').dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await vi.waitFor(() => expect(q('vlm-delete-error')).not.toBeNull());
    expect(q('vlm-delete-error')?.textContent).toBe(
      'local_vlm is active in other projects.',
    );
    expect(q('vlm-delete-projects')?.textContent).toContain('alpha, beta');
    const del = requests.find((r) => r.method === 'DELETE')!;
    expect(del.url).toContain('/vlm/endpoints/local_vlm?expected_revision=3');
  });
});
