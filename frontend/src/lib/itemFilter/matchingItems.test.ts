import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/api_cropRead', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_cropRead')>('$lib/api_cropRead');
  return { ...actual, getCrops: vi.fn() };
});

import { getCrops } from '$lib/api_cropRead';
import { createMatchingItems } from './matchingItems.svelte';
import type { Crop } from '$lib/types';

const crop = (id: string) => ({ id }) as unknown as Crop;

afterEach(() => vi.clearAllMocks());

describe('createMatchingItems', () => {
  it('sends the whole filter, open-vocabulary pair included, on every page', async () => {
    vi.mocked(getCrops).mockImplementation(async (_f, cursor) => ({
      items: [crop(cursor === null ? 'p1' : 'p2')],
      total: 2,
      next_cursor: cursor === null ? 'c2' : null,
      limit: 1,
    }));
    const m = createMatchingItems(() => ({
      class_name: ['widget'],
      open_vocab_set: 'tags',
      source_prompt: 'blue widget',
    }));
    await m.pager.loadFirst();
    await m.pager.loadMore();
    const filter = {
      class_name: ['widget'],
      open_vocab_set: 'tags',
      source_prompt: 'blue widget',
      limit: 48,
    };
    expect(vi.mocked(getCrops).mock.calls.map(([f, cursor]) => [f, cursor])).toEqual([
      [filter, null],
      [filter, 'c2'],
    ]);
    expect(m.pager.items.map((c) => c.id)).toEqual(['p1', 'p2']);
    expect(m.pager.total).toBe(2);
  });

  it('reads the filter at fetch time, so a changed filter reloads with the new one', async () => {
    vi.mocked(getCrops).mockResolvedValue({
      items: [],
      total: 0,
      next_cursor: null,
      limit: 48,
    });
    let q: Record<string, unknown> = { class_name: ['a'] };
    const m = createMatchingItems(() => q);
    await m.pager.loadFirst();
    q = { class_name: ['b'], origin: ['sam3'] };
    await m.pager.loadFirst();
    expect(vi.mocked(getCrops).mock.calls[1]![0]).toMatchObject({
      class_name: ['b'],
      origin: ['sam3'],
    });
  });
});
