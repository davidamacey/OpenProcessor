/**
 * Region rows are identified by the served `row_key`
 * (`<crop_id>#<region_box_id>` or `<crop_id>#item`), not by a client-built
 * pair: the pager's dedup across pages must follow it.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getRegions: vi.fn(), getRegionClusters: vi.fn() };
});
import { getRegions } from '$lib/api';
import { createSlotGalleryController } from './slotGalleryController.svelte';

const row = (crop_id: string, region_box_id: string | null, row_key: string) =>
  ({ crop_id, region_box_id, row_key, id: crop_id }) as never;

afterEach(() => {
  vi.mocked(getRegions).mockReset();
});

describe('region gallery rows are keyed by the served row_key', () => {
  it('drops a repeated row_key on a later page but keeps a different one', async () => {
    vi.mocked(getRegions)
      .mockResolvedValueOnce({
        items: [row('c1', 'b1', 'k-first')],
        total: 4,
      } as never)
      .mockResolvedValueOnce({
        items: [
          row('c1', 'b1', 'k-first'),
          // Same crop and box as the first row, a different served key.
          row('c1', 'b1', 'k-other'),
          row('c2', null, 'c2#item'),
        ],
        total: 4,
      } as never);
    const gallery = createSlotGalleryController(widgetTagSlot);
    await gallery.pager.loadFirst();
    await gallery.pager.loadMore();
    expect(gallery.pager.items.map((p) => (p as { row_key: string }).row_key)).toEqual([
      'k-first',
      'k-other',
      'c2#item',
    ]);
  });
});
