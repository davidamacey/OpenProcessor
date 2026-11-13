import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/api_images', () => ({ listImages: vi.fn() }));
import { listImages } from '$lib/api_images';
import { imageCard, imagePage } from '$lib/test/fixtures/imagePage';
import type { NearDupFilter } from '$lib/types_images';
import { toItemSelection } from '$lib/workspace/selection';
import { createImageGalleryController } from './imageGalleryController.svelte';

const mocked = vi.mocked(listImages);

function make(
  sort: string | null = null,
  nearDup: () => NearDupFilter | null = () => null,
) {
  return createImageGalleryController({
    itemQuery: () => ({ class_name: ['widget'] }),
    sortId: () => sort,
    nearDup,
  });
}

beforeEach(() => mocked.mockReset());

describe('createImageGalleryController', () => {
  it('keeps the served order', async () => {
    mocked.mockResolvedValue(
      imagePage([
        imageCard({ image_id: 'z' }),
        imageCard({ image_id: 'a' }),
        imageCard({ image_id: 'm' }),
      ]),
    );
    const g = make();
    await g.loadFirst();
    expect(g.cards.map((c) => c.image_id)).toEqual(['z', 'a', 'm']);
    expect(g.status).toBe('ready');
  });

  it('appends by next_cursor', async () => {
    mocked.mockResolvedValueOnce(
      imagePage([imageCard({ image_id: 'a' })], { next_cursor: 'c1', total: 2 }),
    );
    mocked.mockResolvedValueOnce(imagePage([imageCard({ image_id: 'b' })], { total: 2 }));
    const g = make();
    await g.loadFirst();
    expect(g.hasMore).toBe(true);
    await g.loadMore();
    expect(mocked.mock.calls[1]![1]).toBe('c1');
    expect(g.cards.map((c) => c.image_id)).toEqual(['a', 'b']);
    expect(g.hasMore).toBe(false);
  });

  it('shows the served empty message', async () => {
    mocked.mockResolvedValue(
      imagePage([], {
        empty_reason: 'no_match',
        empty_message: 'No images match this filter.',
      }),
    );
    const g = make();
    await g.loadFirst();
    expect(g.status).toBe('empty');
    expect(g.emptyMessage).toBe('No images match this filter.');
  });

  it('selection of images builds a filter with image_ids', async () => {
    mocked.mockResolvedValue(
      imagePage([imageCard({ image_id: 'a' }), imageCard({ image_id: 'b' })]),
    );
    const g = make();
    await g.loadFirst();
    g.select('a', undefined);
    g.select('b', { ctrlKey: true } as MouseEvent);
    expect(toItemSelection(g.workspaceSelection, { class_names: ['widget'] })).toEqual({
      filter: { class_names: ['widget'], image_ids: ['a', 'b'] },
    });
  });

  it('sends the unlabelled chip as label_state and the served sort', async () => {
    mocked.mockResolvedValue(imagePage([imageCard()]));
    const g = make('indexed_at:asc');
    g.unlabeledOnly = true;
    await g.loadFirst();
    expect(mocked.mock.calls[0]![0]).toEqual({
      class_name: ['widget'],
      label_state: 'has_unlabeled',
      sort: 'indexed_at:asc',
    });
    expect(mocked.mock.calls[0]![1]).toBeNull();
  });

  it('sends the served near-duplicate filter from the URL and reloads on it', async () => {
    mocked.mockResolvedValue(imagePage([imageCard({ image_id: 'a' })]));
    let role = $state<NearDupFilter | null>(null);
    const g = make(null, () => role);
    const before = g.queryKey;
    role = 'collapsed';
    expect(g.queryKey).not.toBe(before);
    await g.loadFirst();
    expect(mocked.mock.calls[0]![0]).toMatchObject({ near_dup: 'collapsed' });
    role = null;
    await g.loadFirst();
    expect(mocked.mock.calls[1]![0]).not.toHaveProperty('near_dup');
  });
});
