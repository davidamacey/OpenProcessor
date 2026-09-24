/**
 * Structured FastAPI error bodies: the per-run strategy 422 must be
 * recognizable (so the operator sees the valid ids) and any
 * `{detail: {error}}` body must still yield readable text.
 */
import { describe, expect, it } from 'vitest';
import { ApiError, formatValidationEntry, unknownStrategyDetail } from './api';

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

  it('p1 (2026-09-24 interactive pass): prefixes the field name and reworks the raw pydantic pattern phrasing', () => {
    const e = new ApiError(422, '/curation/classes', nameValidationBody);
    expect(e.detail).toBe('name: must match pattern ^[a-z0-9_]+$');
    expect(e.message).toContain('name: must match pattern ^[a-z0-9_]+$');
    // Never the raw pydantic sentence verbatim — the field context and
    // reworded phrasing are the fix.
    expect(e.detail).not.toContain('String should match pattern');
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

describe('formatValidationEntry (p1, 2026-09-24 interactive pass)', () => {
  it('prefixes the last string loc segment as the field name', () => {
    expect(formatValidationEntry({ loc: ['body', 'name'], msg: 'field required' })).toBe(
      'name: field required',
    );
  });

  it('skips "body"/"query" loc segments to find the real field', () => {
    expect(formatValidationEntry({ loc: ['query', 'class_id'], msg: 'invalid' })).toBe(
      'class_id: invalid',
    );
  });

  it('reworks "String should match pattern \'X\'" into "must match pattern X"', () => {
    expect(
      formatValidationEntry({
        loc: ['body', 'name'],
        msg: "String should match pattern '^[a-z0-9_]+$'",
      }),
    ).toBe('name: must match pattern ^[a-z0-9_]+$');
  });

  it('passes an unrecognized message through unreworded', () => {
    expect(
      formatValidationEntry({ loc: ['body', 'x'], msg: 'some other pydantic error' }),
    ).toBe('x: some other pydantic error');
  });

  it('falls back to the bare message when loc has no field segment', () => {
    expect(formatValidationEntry({ loc: ['body'], msg: 'bare message' })).toBe(
      'bare message',
    );
    expect(formatValidationEntry({ msg: 'no loc at all' })).toBe('no loc at all');
  });

  it('returns null for a non-object entry or a missing/non-string msg', () => {
    expect(formatValidationEntry(null)).toBeNull();
    expect(formatValidationEntry('a string')).toBeNull();
    expect(formatValidationEntry({ loc: ['body', 'x'] })).toBeNull();
  });
});
