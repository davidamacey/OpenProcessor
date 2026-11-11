import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import {
  decideNearDup,
  getNearDupGroup,
  previewItemsForImages,
  undecideNearDup,
} from '$lib/api_nearDup';
import { createNearDupGroupController } from './nearDupGroupController.svelte';
import type { NearDupDecisionView, NearDupGroupDetail } from '$lib/types_nearDup';

vi.mock('$lib/api_nearDup', () => ({
  decideNearDup: vi.fn(),
  getNearDupGroup: vi.fn(),
  previewItemsForImages: vi.fn(),
  undecideNearDup: vi.fn(),
}));

const detail: NearDupGroupDetail = {
  group: {
    group_id: 'g1',
    purpose: 'dedup',
    size: 2,
    representative_id: 'a',
    min_edge_cosine: 0.99,
    min_cosine_to_rep: 0.99,
    has_holdout: false,
  },
  members: [
    { member_id: 'a', is_representative: true, cosine_to_rep: 1 },
    { member_id: 'b', is_representative: false, cosine_to_rep: 0.99 },
  ],
};
const made: NearDupDecisionView = {
  id: 'd1',
  level: 'image',
  a_id: 'a',
  b_id: 'b',
  decision: 'different',
  decided_by: 'me',
  decided_at: '2026-01-01T00:00:00Z',
};

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getNearDupGroup).mockResolvedValue(detail);
  vi.mocked(previewItemsForImages).mockResolvedValue({ a: 'ia', b: 'ib' });
});

describe('createNearDupGroupController', () => {
  it('loads the members and, at the image level, their preview items', async () => {
    const c = createNearDupGroupController('image', 'g1');
    await c.load();
    expect(getNearDupGroup).toHaveBeenCalledWith('image', 'g1', expect.anything());
    expect(c.detail?.members).toHaveLength(2);
    expect(previewItemsForImages).toHaveBeenCalledWith(['a', 'b'], expect.anything());
    expect(c.previews).toEqual({ a: 'ia', b: 'ib' });
    expect(c.loading).toBe(false);
    c.dispose();
  });

  it('names a failed read and loads no previews at the item level', async () => {
    vi.mocked(getNearDupGroup).mockRejectedValue(
      new ApiError(404, '/x', { detail: { error: 'group_not_found', message: 'gone' } }),
    );
    const c = createNearDupGroupController('item', 'g1');
    await c.load();
    expect(c.error).toBe('gone');
    expect(c.detail).toBeNull();
    expect(previewItemsForImages).not.toHaveBeenCalled();
    c.dispose();
  });

  it('records a decision and takes it back with its id', async () => {
    vi.mocked(decideNearDup).mockResolvedValue(made);
    vi.mocked(undecideNearDup).mockResolvedValue();
    const c = createNearDupGroupController('image', 'g1');
    await c.load();
    await c.decide('a', 'b', 'different');
    expect(decideNearDup).toHaveBeenCalledWith({
      level: 'image',
      a_id: 'a',
      b_id: 'b',
      decision: 'different',
    });
    expect(c.decisions).toEqual([made]);
    await c.undo('d1');
    expect(undecideNearDup).toHaveBeenCalledWith('d1');
    expect(c.decisions).toEqual([]);
    c.dispose();
  });

  it('keeps no decision and shows the served error when the write is refused', async () => {
    vi.mocked(decideNearDup).mockRejectedValue(
      new ApiError(422, '/x', { detail: 'bad pair' }),
    );
    const c = createNearDupGroupController('image', 'g1');
    await c.decide('a', 'b', 'same');
    expect(c.decisions).toEqual([]);
    expect(c.actionError).toContain('bad pair');
    expect(c.busy).toBe(false);
    c.dispose();
  });
});
