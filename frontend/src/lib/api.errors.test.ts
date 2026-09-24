/**
 * Structured FastAPI error bodies: the per-run strategy 422 must be
 * recognizable (so the operator sees the valid ids) and any
 * `{detail: {error}}` body must still yield readable text.
 */
import { describe, expect, it } from 'vitest';
import { ApiError, unknownStrategyDetail } from './api';

const body422 = {
  detail: {
    error: "unknown prompt_pack 'nope'",
    axis: 'prompt_pack',
    requested: 'nope',
    valid_ids: ['generic_item_v1', 'vehicle_plate_v1'],
  },
};

describe('structured API errors', () => {
  it('parses the unknown-strategy 422', () => {
    const e = new ApiError(422, '/curation/pipeline/auto_label/start', body422);
    expect(unknownStrategyDetail(e)).toEqual({
      axis: 'prompt_pack',
      requested: 'nope',
      valid_ids: ['generic_item_v1', 'vehicle_plate_v1'],
    });
  });

  it('ignores other statuses and plain-string details', () => {
    expect(unknownStrategyDetail(new ApiError(409, '/x', body422))).toBeNull();
    expect(unknownStrategyDetail(new ApiError(422, '/x', { detail: 'bad' }))).toBeNull();
    expect(unknownStrategyDetail(new Error('x'))).toBeNull();
  });

  it("reads a structured detail's error text into the message", () => {
    const e = new ApiError(422, '/x', body422);
    expect(e.detail).toBe("unknown prompt_pack 'nope'");
    expect(e.message).toContain("unknown prompt_pack 'nope'");
  });
});
