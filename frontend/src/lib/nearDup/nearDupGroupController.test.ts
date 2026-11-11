import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import { decideNearDup, getNearDupGroup, undecideNearDup } from '$lib/api_nearDup';
import { createNearDupGroupController } from './nearDupGroupController.svelte';
import type {
  NearDupDecisionView,
  NearDupGroupDetail,
  NearDupMemberView,
} from '$lib/types_nearDup';

vi.mock('$lib/api_nearDup', () => ({
  decideNearDup: vi.fn(),
  getNearDupGroup: vi.fn(),
  undecideNearDup: vi.fn(),
}));

const member = (id: string, rep: boolean): NearDupMemberView => ({
  member_id: id,
  is_representative: rep,
  cosine_to_rep: rep ? 1 : 0.99,
  thumbnail_url: `/p/images/${id}/thumbnail`,
  image_id: id,
  crop_id: null,
  class_name: null,
  in_holdout: false,
  decision: null,
});

const detail: NearDupGroupDetail = {
  group: {
    group_id: 'g1',
    purpose: 'dedup',
    size: 2,
    representative_id: 'a',
    min_edge_cosine: 0.99,
    min_cosine_to_rep: 0.99,
    has_holdout: false,
    representative_thumbnail_url: '/p/images/a/thumbnail',
  },
  members: [member('a', true), member('b', false)],
};
const pinned: NearDupGroupDetail = {
  ...detail,
  members: [
    member('a', true),
    { ...member('b', false), decision: { id: 'd1', kind: 'different' } },
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
  a_thumbnail_url: '/p/images/a/thumbnail',
  b_thumbnail_url: '/p/images/b/thumbnail',
};

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getNearDupGroup).mockResolvedValue(detail);
});

describe('createNearDupGroupController', () => {
  it('loads the members with their served pictures and pins', async () => {
    const c = createNearDupGroupController('image', 'g1');
    await c.load();
    expect(getNearDupGroup).toHaveBeenCalledWith('image', 'g1', expect.anything());
    expect(c.detail?.members).toHaveLength(2);
    expect(c.detail?.members[0]!.thumbnail_url).toBe('/p/images/a/thumbnail');
    expect(c.loading).toBe(false);
    c.dispose();
  });

  it('names a failed read', async () => {
    vi.mocked(getNearDupGroup).mockRejectedValue(
      new ApiError(404, '/x', { detail: { error: 'group_not_found', message: 'gone' } }),
    );
    const c = createNearDupGroupController('item', 'g1');
    await c.load();
    expect(c.error).toBe('gone');
    expect(c.detail).toBeNull();
    c.dispose();
  });

  it('records a decision, reads the served pin, and takes any pin back by its served id', async () => {
    vi.mocked(decideNearDup).mockResolvedValue(made);
    vi.mocked(undecideNearDup).mockResolvedValue();
    const c = createNearDupGroupController('image', 'g1');
    await c.load();
    expect(c.pinned).toBe(0);
    vi.mocked(getNearDupGroup).mockResolvedValue(pinned);
    await c.decide('a', 'b', 'different');
    expect(decideNearDup).toHaveBeenCalledWith({
      level: 'image',
      a_id: 'a',
      b_id: 'b',
      decision: 'different',
    });
    expect(c.pinned).toBe(1);
    expect(c.detail?.members[1]!.decision).toEqual({ id: 'd1', kind: 'different' });
    vi.mocked(getNearDupGroup).mockResolvedValue(detail);
    await c.undo('d1');
    expect(undecideNearDup).toHaveBeenCalledWith('d1');
    expect(c.pinned).toBe(0);
    c.dispose();
  });

  it('shows a pin made in an earlier visit, with no decision made in this one', async () => {
    vi.mocked(getNearDupGroup).mockResolvedValue(pinned);
    const c = createNearDupGroupController('image', 'g1');
    await c.load();
    expect(c.pinned).toBe(1);
    c.dispose();
  });

  it('keeps no decision and shows the served error when the write is refused', async () => {
    vi.mocked(decideNearDup).mockRejectedValue(
      new ApiError(422, '/x', { detail: 'bad pair' }),
    );
    const c = createNearDupGroupController('image', 'g1');
    await c.decide('a', 'b', 'same');
    expect(c.actionError).toContain('bad pair');
    expect(c.busy).toBe(false);
    c.dispose();
  });
});
