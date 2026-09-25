/**
 * Mount test for `CropMetaPanel`'s history and timestamp rendering
 * (visual audit 2026-09-24, K7): history rows used to omit the resulting
 * class when a write left none ("vlm_label_batch (coco_yolo11_proposal)")
 * and every timestamp was raw ISO with microseconds.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { Crop } from '$lib/types';

vi.mock('$lib/api', () => ({
  getCropHistory: vi.fn(async () => ({
    crop_id: 'c1',
    entries: [
      {
        writer: 'vlm_label_batch',
        at: '2026-09-24T16:26:50.123456+00:00',
        class_name: null,
        class_source: 'coco_yolo11_proposal',
      },
      {
        writer: 'vlm_label_batch',
        at: '2026-09-24T16:26:57.832024+00:00',
        class_name: 'class_b',
        class_source: 'vlm',
      },
    ],
  })),
  getCropContext: vi.fn(async () => ({ image: null, items: [] })),
  getThumbUrl: (id: string) => `/thumb/${id}`,
}));

const { default: CropMetaPanel } = await import('./CropMetaPanel.svelte');

function crop(): Crop {
  return {
    id: 'c1',
    class_id: 64,
    class_name: 'class_b',
    class_source: 'vlm',
    label_source: 'vlm',
    class_labeled_at: '2026-09-24T16:26:57.832024+00:00',
    updated_at: '2026-09-24T17:54:10.220693+00:00',
  } as unknown as Crop;
}

let instance: unknown;
let target: HTMLDivElement;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

async function render(): Promise<HTMLDivElement> {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CropMetaPanel, { target, props: { crop: crop() } });
  flushSync();
  await new Promise((r) => setTimeout(r, 0));
  flushSync();
  return target;
}

describe('CropMetaPanel history (K7)', () => {
  it('names the resulting class on every row, "no class" when the write left none', async () => {
    const el = await render();
    const rows = [...el.querySelectorAll('[data-testid="history-entry"]')].map((r) =>
      (r.textContent ?? '').replace(/\s+/g, ' ').trim(),
    );
    expect(rows).toHaveLength(2);
    expect(rows[0]).toContain('→ no class');
    expect(rows[0]).toContain('source: coco_yolo11_proposal');
    expect(rows[1]).toContain('→ class_b');
  });

  it('formats timestamps instead of printing raw ISO with microseconds', async () => {
    const el = await render();
    const text = el.textContent ?? '';
    expect(text).toContain('2026-09-24 16:26:57 UTC');
    expect(text).not.toContain('.832024');
    expect(text).not.toContain('.123456');
  });
});
