import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import type { SeedFromDetectorResponse } from '$lib/types_detector';
import { SeedFromDetector, type SeedDeps } from './seedController.svelte';

const DRY: SeedFromDetectorResponse = {
  dry_run: true,
  detector_model: 'm',
  created: [
    { class_id: null, name: 'widget', detector_label: 'widget' },
    { class_id: null, name: 'tag_label', detector_label: 'tag label' },
  ],
  skipped: [{ name: 'gadget', detector_label: 'gadget', reason: 'exists' }],
  conflicts: [{ detector_label: '??', class_id_in_detector: 7, reason: 'unnamed_label' }],
};

function deps(over: Partial<SeedDeps> = {}): SeedDeps {
  return {
    getConfig: vi.fn(async () => ({
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
    })),
    seed: vi.fn(async (req) => ({
      ...DRY,
      dry_run: req.dry_run,
      created: DRY.created.map((c, i) => ({ ...c, class_id: 10 + i })),
    })),
    onSeeded: vi.fn(async () => {}),
    ...over,
  };
}

describe('SeedFromDetector', () => {
  it('lists the detector labels; no detector means no panel', async () => {
    const s = new SeedFromDetector(deps());
    await s.load();
    expect(s.labelNames).toEqual(['widget', 'tag label']);
    expect(s.available).toBe(true);

    const none = new SeedFromDetector(
      deps({ getConfig: vi.fn(async () => ({ detector: null })) }),
    );
    await none.load();
    expect(none.available).toBe(false);
  });

  it('preview is a dry run and omits names when none are chosen', async () => {
    const d = deps();
    const s = new SeedFromDetector(d);
    await s.load();
    await s.preview();
    expect(d.seed).toHaveBeenCalledTimes(1);
    expect(vi.mocked(d.seed).mock.calls[0]![0]).toEqual({ dry_run: true });
    expect(s.result?.created).toHaveLength(2);
  });

  it('sends the chosen names, and create is never sent before a preview', async () => {
    const d = deps();
    const s = new SeedFromDetector(d);
    await s.load();
    await s.create();
    expect(d.seed).not.toHaveBeenCalled();
    s.chosen = ['widget'];
    await s.preview();
    expect(vi.mocked(d.seed).mock.calls[0]![0]).toEqual({
      dry_run: true,
      names: ['widget'],
    });
  });

  it('create sends dry_run false with the previewed names, then reloads classes', async () => {
    const d = deps();
    const s = new SeedFromDetector(d);
    await s.load();
    s.chosen = ['widget'];
    await s.preview();
    await s.create();
    expect(vi.mocked(d.seed).mock.calls[1]![0]).toEqual({
      dry_run: false,
      names: ['widget'],
    });
    expect(d.onSeeded).toHaveBeenCalledTimes(1);
    expect(s.created?.created.map((c) => c.class_id)).toEqual([10, 11]);
    expect(s.result).toBeNull();
  });

  it('changing the choice drops a stale preview', async () => {
    const s = new SeedFromDetector(deps());
    await s.load();
    await s.preview();
    s.setChosen(['tag label']);
    expect(s.result).toBeNull();
  });

  it('shows the served 503 and 422 refusals', async () => {
    const s = new SeedFromDetector(
      deps({
        seed: vi.fn(async () => {
          throw new ApiError(422, 'u', {
            detail: {
              error: 'unknown_detector_names',
              message: 'Not detector labels.',
              unknown_names: ['gizmo'],
            },
          });
        }),
      }),
    );
    await s.load();
    await s.preview();
    expect(s.errorLines).toEqual(['Not detector labels.', 'Unknown: gizmo']);
    expect(s.result).toBeNull();
  });
});
