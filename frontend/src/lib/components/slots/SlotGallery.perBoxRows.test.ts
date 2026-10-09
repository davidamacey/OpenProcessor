/**
 * `GET /regions` returns one ROW per box (OpenProcessor `RegionRow`: key a
 * row by its served `row_key`), so one item with three boxes is three
 * rows sharing a crop_id. Keying the grid by crop_id alone threw
 * each_key_duplicate and blanked the gallery. Mounts the real component.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import SlotGallery from './SlotGallery.svelte';
import { createSlotGalleryController } from '../../../routes/p/[project]/clusters/slotGalleryController.svelte';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getRegionClusters: vi.fn(), getRegions: vi.fn() };
});
import { getRegions } from '$lib/api';

function row(cropId: string, boxId: string | null, thumb: string) {
  return {
    crop_id: cropId,
    id: cropId,
    image_path: '/x.jpg',
    bbox_norm: [0, 0, 1, 1],
    region_box_id: boxId,
    row_key: `${cropId}#${boxId ?? 'item'}`,
    thumbnail_url: thumb,
    updated_at: '2026-10-03T00:00:00Z',
  };
}

let target: HTMLDivElement;
let instance: unknown;

vi.stubGlobal(
  'IntersectionObserver',
  class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  },
);

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

async function render(items: unknown[]): Promise<HTMLElement> {
  vi.mocked(getRegions).mockResolvedValue({
    items,
    total: items.length,
    total_rows: items.length,
  } as never);
  const gallery = createSlotGalleryController(widgetTagSlot);
  await gallery.loadFirst();
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SlotGallery, { target, props: { gallery } } as never);
  flushSync();
  return target;
}

describe('SlotGallery per-box rows', () => {
  it('renders every box row of one item (same crop_id, distinct region_box_id)', async () => {
    const el = await render([
      row('c1', 'b1', '/t/b1'),
      row('c1', 'b2', '/t/b2'),
      row('c1', 'b3', '/t/b3'),
    ]);
    expect(el.querySelectorAll('img').length).toBe(3);
  });

  it('keeps the box rows of an item the first page already returned (pager dedup is per box)', async () => {
    vi.mocked(getRegions)
      .mockResolvedValueOnce({
        items: [row('c1', 'b1', '/t/b1')],
        total: 2,
        total_rows: 2,
      } as never)
      .mockResolvedValueOnce({
        items: [row('c1', 'b2', '/t/b2')],
        total: 2,
        total_rows: 2,
      } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.loadFirst();
    await gallery.pager.loadMore();
    expect(gallery.pager.items.length).toBe(2);
  });

  it('control: rows of distinct items still render', async () => {
    const el = await render([row('c1', 'b1', '/t/b1'), row('c2', 'b1', '/t/b2')]);
    expect(el.querySelectorAll('img').length).toBe(2);
  });
});
