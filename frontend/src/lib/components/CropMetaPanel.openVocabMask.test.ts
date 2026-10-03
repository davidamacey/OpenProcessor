/**
 * `<CropMetaPanel>` draws the served `mask_polygon` over the source image
 * (mount-based): one outline, in the shape the server returned it, only
 * when the shown crop carries one. A list-served crop (mask null) shows
 * none, and nothing is fetched just for the mask.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { Crop } from '$lib/types';
import CropMetaPanel from './CropMetaPanel.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;
let urls: string[];

const MASK = [
  [0.1, 0.1],
  [0.5, 0.1],
  [0.5, 0.6],
  [0.1, 0.6],
];

beforeEach(() => {
  urls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const u = String(url);
      urls.push(u);
      const body = u.endsWith('/context')
        ? { image: { width: 800, height: 600, source: 'src' }, items: [] }
        : u.endsWith('/history')
          ? { crop_id: 'crop-1', entries: [] }
          : {};
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    }),
  );
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  vi.unstubAllGlobals();
});

async function render(over: Partial<Crop>) {
  const crop = {
    id: 'crop-1',
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.5, h: 0.5 },
    class_id: 3,
    class_name: 'widget_a',
    cluster_id: 7,
    label_confidence: 0.9,
    label_source: 'model',
    class_source: 'model',
    updated_at: '',
    slots: {},
    ...over,
  } as unknown as Crop;
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CropMetaPanel, { target, props: { crop } } as never);
  await vi.waitFor(() =>
    expect(urls.some((u) => u.endsWith('/crops/crop-1/context'))).toBe(true),
  );
  await new Promise((r) => setTimeout(r, 0));
  flushSync();
}

const polys = () => target.querySelectorAll('[data-testid="overlay-extra-polygon"]');

describe('CropMetaPanel open-vocabulary mask', () => {
  it('draws one outline from the served mask polygon, untouched', async () => {
    await render({ mask_polygon: MASK });
    await vi.waitFor(() => expect(polys()).toHaveLength(1));
    expect(polys()[0]!.getAttribute('points')).toBe('0.1,0.1 0.5,0.1 0.5,0.6 0.1,0.6');
    expect(polys()[0]!.getAttribute('data-dimmed')).toBe('false');
    expect(polys()[0]!.querySelector('title')!.textContent).toContain('mask');
  });

  it('draws nothing when the shown crop has no mask, and does not fetch for one', async () => {
    await render({ mask_polygon: null });
    expect(polys()).toHaveLength(0);
    expect(urls.every((u) => !u.includes('mask'))).toBe(true);
    expect(urls.filter((u) => u.endsWith('/crops/crop-1'))).toHaveLength(0);
  });
});
