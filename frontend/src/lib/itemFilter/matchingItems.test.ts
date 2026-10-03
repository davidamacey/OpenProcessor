import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getCrops: vi.fn() };
});

import { getCrops } from '$lib/api';
import { createMatchingItems } from './matchingItems.svelte';
import type { Crop } from '$lib/types';

const crop = (id: string) => ({ id }) as unknown as Crop;

afterEach(() => vi.clearAllMocks());

describe('createMatchingItems', () => {
  it('sends the whole filter, open-vocabulary pair included, on every page', async () => {
    vi.mocked(getCrops).mockImplementation(async (f = {}) => ({
      items: [crop(`p${f.page}`)],
      total: 2,
      page: f.page ?? 1,
      page_size: 1,
    }));
    const m = createMatchingItems(() => ({
      class_name: ['widget'],
      open_vocab_set: 'tags',
      source_prompt: 'blue widget',
    }));
    await m.pager.loadFirst();
    await m.pager.loadMore();
    expect(vi.mocked(getCrops).mock.calls.map(([f]) => f)).toEqual([
      {
        class_name: ['widget'],
        open_vocab_set: 'tags',
        source_prompt: 'blue widget',
        page: 1,
        limit: 48,
      },
      {
        class_name: ['widget'],
        open_vocab_set: 'tags',
        source_prompt: 'blue widget',
        page: 2,
        limit: 48,
      },
    ]);
    expect(m.pager.items.map((c) => c.id)).toEqual(['p1', 'p2']);
    expect(m.pager.total).toBe(2);
  });

  it('reads the filter at fetch time, so a changed filter reloads with the new one', async () => {
    vi.mocked(getCrops).mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 48,
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
