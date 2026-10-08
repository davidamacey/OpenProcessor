/**
 * V-4: TrainForm's first preflight spec selects only the classes the
 * server doesn't report as short of data (served `trainable_gap`), so a
 * 0-crop class doesn't block preflight out of the box.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import TrainForm from './TrainForm.svelte';
import { classesStore } from '$stores/classes.svelte';
import type { RegistryClass } from '$lib/types';
import type { TrainJobSpec } from '$lib/types_train';

function cls(id: number, gap: number): RegistryClass {
  return {
    id,
    name: `widget_${id}`,
    group: null,
    count: 0,
    validated_count: 0,
    cluster_size: 0,
    added_at: '2026-01-01T00:00:00Z',
    adequacy: 'block',
    kind: 'item',
    trainable: 0,
    trainable_gap: gap,
  };
}

let target: HTMLDivElement;
let instance: unknown;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  classesStore.classes = [];
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('TrainForm default class selection (V-4)', () => {
  it('first preflight includes only classes with no served trainable_gap', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(
        async () =>
          new Response(JSON.stringify({ options: [], allowed_ids: [], presets: [] }), {
            status: 200,
            headers: { 'content-type': 'application/json' },
          }),
      ),
    );
    classesStore.classes = [cls(0, 0), cls(1, 0), cls(2, 20)];
    const onPreflight = vi.fn<(spec: TrainJobSpec) => void>();
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(TrainForm, {
      target,
      props: {
        datasetExportDir: '/exports/current',
        profiles: [],
        presets: [],
        preflight: null,
        preflighting: false,
        starting: false,
        onPreflight,
        onStart: () => {},
        onStartCampaign: () => {},
      },
    });
    flushSync();
    await vi.waitFor(() => expect(onPreflight).toHaveBeenCalled(), { timeout: 2000 });
    const spec = onPreflight.mock.calls.at(-1)![0];
    expect(spec.include_classes).toEqual([0, 1]);
  });

  it('F-63(c): every form control has an accessible label (hyperparameters expanded)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(
        async () =>
          new Response(JSON.stringify({ options: [], allowed_ids: [], presets: [] }), {
            status: 200,
            headers: { 'content-type': 'application/json' },
          }),
      ),
    );
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(TrainForm, {
      target,
      props: {
        datasetExportDir: '/exports/current',
        profiles: [],
        presets: [],
        preflight: null,
        preflighting: false,
        starting: false,
        onPreflight: () => {},
        onStart: () => {},
        onStartCampaign: () => {},
      },
    });
    flushSync();
    [...target.querySelectorAll('button')]
      .find((b) => b.textContent?.includes('Hyperparameters'))
      ?.click();
    flushSync();
    const controls = [...target.querySelectorAll('input, select, textarea')];
    expect(controls.length).toBeGreaterThan(5);
    const unlabeled = controls
      .filter(
        (el) =>
          !(el as HTMLInputElement).labels?.length && !el.getAttribute('aria-label'),
      )
      .map((el) => el.outerHTML.slice(0, 120));
    expect(unlabeled).toEqual([]);
  });
});
