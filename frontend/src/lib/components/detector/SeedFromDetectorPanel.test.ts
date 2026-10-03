import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import SeedFromDetectorPanel from './SeedFromDetectorPanel.svelte';
import { SeedFromDetector, type SeedDeps } from '$lib/detector/seedController.svelte';
import type { SeedFromDetectorResponse } from '$lib/types_detector';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  document.body.innerHTML = '';
});

const DRY: SeedFromDetectorResponse = {
  dry_run: true,
  detector_model: 'widget_detector_v1',
  created: [{ class_id: null, name: 'tag_label', detector_label: 'tag label' }],
  skipped: [{ name: 'widget', detector_label: 'widget', reason: 'exists' }],
  conflicts: [{ detector_label: 'x', class_id_in_detector: 5, reason: 'duplicate_slug' }],
};

function render(over: Partial<SeedDeps> = {}) {
  const deps: SeedDeps = {
    getConfig: async () => ({
      detector: {
        model: 'm',
        version: '1',
        input_size: 640,
        assigns_class: false,
        confidence_floor_applies: false,
        n_labels: 2,
        labels: [
          { class_id: 0, name: 'widget', slug: 'widget' },
          { class_id: 1, name: 'tag label', slug: 'tag_label' },
        ],
      },
    }),
    seed: vi.fn(async (req) => ({ ...DRY, dry_run: req.dry_run })),
    onSeeded: vi.fn(async () => {}),
    ...over,
  };
  const controller = new SeedFromDetector(deps);
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SeedFromDetectorPanel, { target, props: { controller } });
  flushSync();
  return deps;
}

const q = (id: string) => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);

describe('SeedFromDetectorPanel', () => {
  it('is absent when the deployment reports no detector', async () => {
    render({ getConfig: async () => ({ detector: null }) });
    await new Promise((r) => setTimeout(r, 20));
    flushSync();
    expect(q('seed-panel')).toBeNull();
  });

  it('previews as a dry run and shows the served lists with reasons', async () => {
    const deps = render();
    await vi.waitFor(() => {
      flushSync();
      expect(q('seed-panel')).not.toBeNull();
    });
    q('seed-preview')!.click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('seed-result')).not.toBeNull();
    });
    expect(vi.mocked(deps.seed).mock.calls[0]![0]).toEqual({ dry_run: true });
    expect(q('seed-created')!.textContent).toContain('would create');
    expect(q('seed-created')!.textContent).toContain('tag_label');
    expect(q('seed-created')!.textContent).toContain('tag label');
    expect(q('seed-skipped')!.textContent).toContain('Exists');
    expect(q('seed-conflicts')!.textContent).toContain('Duplicate slug');
    expect(q('seed-conflicts')!.textContent).toContain('5');
  });

  it('creates only after the confirm dialog', async () => {
    const deps = render();
    await vi.waitFor(() => {
      flushSync();
      expect(q('seed-panel')).not.toBeNull();
    });
    q('seed-preview')!.click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('seed-create')).not.toBeNull();
    });
    q('seed-create')!.click();
    flushSync();
    expect(vi.mocked(deps.seed)).toHaveBeenCalledTimes(1);
    const confirm = [...document.querySelectorAll('button')].find(
      (b) => b.textContent?.trim() === 'Create 1 classes' && b !== q('seed-create'),
    )!;
    confirm.click();
    await vi.waitFor(() => expect(deps.seed).toHaveBeenCalledTimes(2));
    expect(vi.mocked(deps.seed).mock.calls[1]![0]).toEqual({ dry_run: false });
    await vi.waitFor(() => expect(deps.onSeeded).toHaveBeenCalledTimes(1));
  });
});
