import { describe, expect, it } from 'vitest';
import { ApiError } from '$lib/api_errors';
import { nearDupStaleLevels, runModeFor } from './staleRefusal';

const body = {
  detail: {
    code: 'near_dup_stale',
    levels: [{ level: 'item', stale_reasons: ['pending'], pending_count: 4 }],
  },
};

describe('nearDupStaleLevels', () => {
  it('reads the stale levels of a 409 near_dup_stale', () => {
    expect(nearDupStaleLevels(new ApiError(409, '/x', body))).toEqual(body.detail.levels);
  });

  it('is null for another 409, another status and a non-ApiError', () => {
    expect(
      nearDupStaleLevels(
        new ApiError(409, '/x', { detail: { code: 'other', levels: [] } }),
      ),
    ).toBeNull();
    expect(nearDupStaleLevels(new ApiError(422, '/x', body))).toBeNull();
    expect(nearDupStaleLevels(new Error('x'))).toBeNull();
    expect(nearDupStaleLevels(new ApiError(409, '/x', { detail: 'plain' }))).toBeNull();
  });
});

describe('runModeFor', () => {
  it('is incremental only when new items are the sole reason', () => {
    expect(
      runModeFor({ level: 'item', stale_reasons: ['pending'], pending_count: 1 }),
    ).toBe('incremental');
    expect(
      runModeFor({
        level: 'item',
        stale_reasons: ['rebuild_required'],
        pending_count: 0,
      }),
    ).toBe('full');
    expect(
      runModeFor({ level: 'image', stale_reasons: ['no_run'], pending_count: 0 }),
    ).toBe('full');
    expect(
      runModeFor({
        level: 'image',
        stale_reasons: ['pending', 'rebuild_required'],
        pending_count: 2,
      }),
    ).toBe('full');
  });
});
