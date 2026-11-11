import { beforeEach, describe, expect, it, vi } from 'vitest';
import { listNearDupDecisions, undecideNearDup } from '$lib/api_nearDup';
import { createNearDupDecisionsController } from './nearDupDecisionsController.svelte';
import type { NearDupDecisionView } from '$lib/types_nearDup';

vi.mock('$lib/api_nearDup', () => ({
  listNearDupDecisions: vi.fn(),
  undecideNearDup: vi.fn(),
}));

const decision = (id: string): NearDupDecisionView => ({
  id,
  level: 'image',
  a_id: `a-${id}`,
  b_id: `b-${id}`,
  decision: 'different',
  decided_by: 'me',
  decided_at: '2026-01-01T00:00:00Z',
  a_thumbnail_url: `/p/images/a-${id}/thumbnail`,
  b_thumbnail_url: `/p/images/b-${id}/thumbnail`,
});

beforeEach(() => vi.resetAllMocks());

describe('createNearDupDecisionsController', () => {
  it('pages the served list with its cursor', async () => {
    vi.mocked(listNearDupDecisions)
      .mockResolvedValueOnce({ items: [decision('2')], next_cursor: 'c1' })
      .mockResolvedValueOnce({ items: [decision('1')], next_cursor: null });
    const c = createNearDupDecisionsController();
    await c.load();
    expect(c.items.map((d) => d.id)).toEqual(['2']);
    expect(c.hasMore).toBe(true);
    await c.load(true);
    expect(vi.mocked(listNearDupDecisions).mock.calls[1]![0]).toMatchObject({
      cursor: 'c1',
    });
    expect(c.items.map((d) => d.id)).toEqual(['2', '1']);
    expect(c.hasMore).toBe(false);
    c.dispose();
  });

  it('undoes any pin by its served id, reads the list again and tells the page', async () => {
    vi.mocked(listNearDupDecisions)
      .mockResolvedValueOnce({ items: [decision('2'), decision('1')], next_cursor: null })
      .mockResolvedValueOnce({ items: [decision('1')], next_cursor: null });
    vi.mocked(undecideNearDup).mockResolvedValue();
    const changed = vi.fn();
    const c = createNearDupDecisionsController(changed);
    await c.load();
    await c.undo('2');
    expect(undecideNearDup).toHaveBeenCalledWith('2', expect.anything());
    expect(c.items.map((d) => d.id)).toEqual(['1']);
    expect(changed).toHaveBeenCalledTimes(1);
    c.dispose();
  });
});
