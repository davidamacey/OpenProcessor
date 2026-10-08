/**
 * `AugmentationPanel` used to hardcode a `PRESETS` id list matching the
 * trainer's own table by hand (`src/lib/contract/augmentPresets.test.ts`
 * diffed it against a local OpenProcessor checkout). OpenProcessor df01309
 * added `GET {API_PREFIX}/train/augmentation_presets`, so the panel now
 * renders whatever the backend serves — this proves the mount actually
 * does that, and that it degrades to a read-only display when the
 * endpoint isn't there yet (the live backend at deploy time, pre-cutover).
 *
 * Mount-based (Svelte 5 mount/unmount/flushSync under jsdom), same
 * harness as `TrainForm.gpuPicker.test.ts`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import AugmentationPanel from './AugmentationPanel.svelte';
import type { AugmentationSpec } from '$lib/types_train';

let target: HTMLDivElement;
let instance: unknown;

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

async function flushMicrotasks(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
}

function mountPanel(value: AugmentationSpec | null): void {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(AugmentationPanel, {
    target,
    props: {
      value,
      setValue: () => {},
    },
  } as never);
}

function expandPanel(): void {
  const toggle = target.querySelector<HTMLButtonElement>('button[aria-expanded]');
  toggle?.click();
  flushSync();
}

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  vi.unstubAllGlobals();
});

describe('AugmentationPanel — served preset list', () => {
  it('renders the served presets (label + description + orientation note), not a hardcoded list', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        presets: [
          {
            id: 'none',
            label: 'None',
            description: 'No changes.',
            orientation_sensitive: false,
          },
          {
            id: 'text_targets',
            label: 'Text targets',
            description: 'Small text-bearing targets.',
            orientation_sensitive: true,
          },
        ],
        default: 'none',
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    mountPanel({
      enabled: true,
      multiplier: 3,
      preset: 'text_targets',
      albumentations: {},
      per_class_multiplier: {},
    });
    flushSync();
    await flushMicrotasks();
    flushSync();
    expandPanel();

    const options = Array.from(
      target.querySelectorAll<HTMLOptionElement>('select option'),
    );
    expect(options.map((o) => o.value)).toEqual(['none', 'text_targets']);
    expect(target.textContent).toContain('Text targets');
    expect(target.textContent).toContain('Small text-bearing targets.');
    expect(target.textContent).toContain('horizontal flip disabled for this preset');
  });

  it('shows the load error and no picker when the preset list fails to load', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ detail: 'presets unavailable' }, 422));
    vi.stubGlobal('fetch', fetchMock);

    mountPanel({
      enabled: true,
      multiplier: 3,
      preset: 'outdoor_scene',
      albumentations: {},
      per_class_multiplier: {},
    });
    flushSync();
    await flushMicrotasks();
    flushSync();
    expandPanel();

    expect(target.querySelector('select')).toBeNull();
    expect(target.textContent).toContain('Could not load presets');
    expect(target.textContent).toContain('presets unavailable');
  });
});
