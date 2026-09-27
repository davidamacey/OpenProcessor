/**
 * `/models` renders the served `optional` + `not_installed` status
 * (OpenProcessor ba88751) as a neutral "optional · not installed" pill,
 * not as a warning, and drops the "protected: in use by the pipeline"
 * chip for a model that isn't installed. Real mount under jsdom, with
 * `fetch` answering `GET {API_PREFIX}/models/status`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import ModelsPage from './+page.svelte';

let target: HTMLDivElement;
let instance: unknown;

const BASE = {
  role: 'region_detector',
  kind: 'triton',
  model_type: 'TensorRT detection',
  version: null,
  inference_count: 0,
  exec_count: 0,
  inference_failed: 0,
  avg_latency_ms: null,
  last_error: null,
  endpoint: 'http://triton:8000',
  is_region_protected: true,
  unloadable: true,
  requires_force_to_unload: false,
};

async function render(models: unknown[]): Promise<void> {
  vi.stubGlobal(
    'fetch',
    vi.fn(
      async () =>
        new Response(JSON.stringify({ models }), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
    ),
  );
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ModelsPage, { target });
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  flushSync();
}

function pills(): string[] {
  return [...target.querySelectorAll('[data-testid="model-status-pill"]')].map(
    (el) => el.textContent?.trim() ?? '',
  );
}

afterEach(() => {
  if (instance) unmount(instance as Parameters<typeof unmount>[0]);
  target?.remove();
  vi.unstubAllGlobals();
});

describe('/models optional + not_installed', () => {
  it('shows a neutral "optional · not installed" pill and no protected chip', async () => {
    await render([
      {
        ...BASE,
        name: 'region_detector_v1',
        friendly_name: 'Region Detector',
        status: 'not_installed',
        optional: true,
      },
    ]);
    expect(pills()).toEqual(['optional · not installed']);
    const pill = target.querySelector('[data-testid="model-status-pill"]')!;
    expect(pill.className).not.toMatch(/yellow|red/);
    expect(target.textContent).not.toMatch(/protected/i);
  });

  it('keeps "not ready" and the protected chip for an installed-but-unloaded model', async () => {
    await render([
      {
        ...BASE,
        name: 'region_detector_v1',
        friendly_name: 'Region Detector',
        status: 'not_ready',
        optional: true,
      },
    ]);
    expect(pills()).toEqual(['not ready']);
    expect(target.textContent).toMatch(/protected/i);
  });

  it('renders an older backend without `optional` exactly as before', async () => {
    await render([
      {
        ...BASE,
        name: 'encoder',
        friendly_name: 'Encoder',
        status: 'ready',
        is_region_protected: false,
      },
    ]);
    expect(pills()).toEqual(['ready']);
  });
});
