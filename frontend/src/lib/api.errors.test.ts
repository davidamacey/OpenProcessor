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

describe('Pydantic validation-error arrays (W4, 2026-09-24)', () => {
  // Live on d037be8: `POST {API_PREFIX}/classes` with a name that fails
  // `^[a-z0-9_]+$` returns `{detail: [{type, loc, msg, input, ctx}]}`, not a
  // plain string. `/classes` and `AddClassModal` dropped their own regex
  // and now depend on this text reaching the toast.
  const nameValidationBody = {
    detail: [
      {
        type: 'string_pattern_mismatch',
        loc: ['body', 'name'],
        msg: "String should match pattern '^[a-z0-9_]+$'",
        input: 'Bad Name!',
        ctx: { pattern: '^[a-z0-9_]+$' },
      },
    ],
  };

  it('extracts the msg field from a single-entry validation array', () => {
    const e = new ApiError(422, '/curation/classes', nameValidationBody);
    expect(e.detail).toBe("String should match pattern '^[a-z0-9_]+$'");
    expect(e.message).toContain("String should match pattern '^[a-z0-9_]+$'");
  });

  it('joins multiple validation entries with "; "', () => {
    const body = {
      detail: [{ msg: 'first problem' }, { msg: 'second problem' }],
    };
    const e = new ApiError(422, '/x', body);
    expect(e.detail).toBe('first problem; second problem');
  });

  it('falls back to null (not a crash) when no entry has a msg string', () => {
    const e = new ApiError(422, '/x', { detail: [{ loc: ['body'] }] });
    expect(e.detail).toBeNull();
  });
});
