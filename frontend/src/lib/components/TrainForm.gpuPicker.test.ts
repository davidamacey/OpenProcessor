/**
 * Mount-based behavior test for TrainForm's GPU picker
 * (docs/design/test-audit-2026-09-24.md P1-4): it must render the served
 * options and preselect the backend's marked default. `getTrainGpus` goes
 * through the real `api.ts` fetch path, mocked at the `fetch` boundary so
 * the assertion covers the actual wiring (effect -> getTrainGpus ->
 * defaultGpuValue -> radio checked state), not a re-implementation.
 *
 * Supersedes the "loads the served options"/"renders one radio per
 * option" scans that used to live in `TrainForm.test.ts` (now trimmed to
 * only the checks a DOM mount can't reach — payload wiring, dead-code
 * absence, prop naming).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import TrainForm from './TrainForm.svelte';

let target: HTMLDivElement;
let instance: unknown;

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

// fetch() -> res.json() -> state update each add their own microtask/tick,
// so a couple of bare Promise.resolve() flushes aren't reliably enough
// (verified empirically against this mock's Response.json() path); a
// zero-delay macrotask settles it.
async function flushMicrotasks(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
}

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  vi.unstubAllGlobals();
});

// TrainForm mounts both the GPU picker (`getTrainGpus`) and
// `AugmentationPanel` (`getAugmentationPresets`) — each hits `fetch`
// independently, so a single shared `Response` (`mockResolvedValue`)
// would have its body consumed by whichever call wins the race, failing
// the other's `res.json()`. Route by URL and return a fresh `Response`
// per call instead.
function routeByUrl(
  gpuBody: unknown,
  augmentationBody: unknown = { presets: [], default: 'balanced_default' },
): (url: string) => Promise<Response> {
  return (url: string) =>
    Promise.resolve(
      url.includes('augmentation_presets')
        ? jsonResponse(augmentationBody)
        : jsonResponse(gpuBody),
    );
}

describe('TrainForm — GPU picker', () => {
  it('renders the served GPU options and preselects the backend default', async () => {
    const fetchMock = vi.fn().mockImplementation(
      routeByUrl({
        options: [
          {
            value: '0',
            label: 'GPU 0 (A6000)',
            advisory: null,
            stops_containers: [],
            default: false,
          },
          {
            value: '1',
            label: 'GPU 1 (3080 Ti)',
            advisory: 'shared with desktop',
            stops_containers: [],
            default: true,
          },
          {
            value: '2',
            label: 'GPU 2 (A6000)',
            advisory: null,
            stops_containers: [],
            default: false,
          },
        ],
        allowed_ids: [0, 1, 2],
        unrestricted: false,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(TrainForm, {
      target,
      props: {
        datasetExportDir: '/data/export',
        profiles: [],
        presets: [],
        preflight: null,
        preflighting: false,
        starting: false,
        onPreflight: () => {},
        onStart: () => {},
        onStartCampaign: () => {},
      },
    } as never);
    flushSync();
    await flushMicrotasks();
    flushSync();

    const radios = Array.from(
      target.querySelectorAll<HTMLInputElement>('input[name="cuda-devices"]'),
    );
    expect(radios.map((r) => r.value)).toEqual(['0', '1', '2']);
    expect(target.textContent).toContain('GPU 1 (3080 Ti)');

    const checked = radios.find((r) => r.checked);
    expect(checked?.value).toBe('1');
    expect(target.textContent).toContain('shared with desktop');
  });

  it('shows a free-text field when the backend is unrestricted', async () => {
    const fetchMock = vi
      .fn()
      .mockImplementation(
        routeByUrl({ options: [], allowed_ids: [], unrestricted: true }),
      );
    vi.stubGlobal('fetch', fetchMock);

    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(TrainForm, {
      target,
      props: {
        datasetExportDir: '/data/export',
        profiles: [],
        presets: [],
        preflight: null,
        preflighting: false,
        starting: false,
        onPreflight: () => {},
        onStart: () => {},
        onStartCampaign: () => {},
      },
    } as never);
    flushSync();
    await flushMicrotasks();
    flushSync();

    expect(target.querySelector('input[aria-label="CUDA visible devices"]')).toBeTruthy();
    expect(target.querySelectorAll('input[name="cuda-devices"]').length).toBe(0);
  });
});
