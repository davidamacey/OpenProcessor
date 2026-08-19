/**
 * Tests for ApiError's message composition.
 *
 * Every UI callsite renders `(e as Error).message`, so the server's reason
 * has to be baked into the message or the operator never sees it.
 */

import { describe, expect, it } from 'vitest';
import { ApiError } from './api';

const URL = 'http://localhost:4603/op/crops/batch_label';

describe('ApiError', () => {
  it("appends a FastAPI 'detail' string to the message", () => {
    const e = new ApiError(422, URL, { detail: 'plate bbox outside crop envelope' });
    expect(e.detail).toBe('plate bbox outside crop envelope');
    expect(e.message).toBe(`API 422 ${URL} — plate bbox outside crop envelope`);
  });

  it("falls back to a 'message' property", () => {
    const e = new ApiError(400, URL, { message: 'hotkey already bound' });
    expect(e.message).toContain('hotkey already bound');
  });

  it('accepts a plain-text body', () => {
    const e = new ApiError(502, URL, 'upstream unavailable');
    expect(e.detail).toBe('upstream unavailable');
  });

  it('truncates a long detail to 200 chars', () => {
    const e = new ApiError(422, URL, { detail: 'x'.repeat(500) });
    expect(e.detail).toHaveLength(200);
    expect(e.detail?.endsWith('…')).toBe(true);
  });

  it('leaves the message unadorned when there is no usable detail', () => {
    expect(new ApiError(500, URL, null).message).toBe(`API 500 ${URL}`);
    expect(new ApiError(500, URL, { detail: '   ' }).message).toBe(`API 500 ${URL}`);
    expect(new ApiError(500, URL, { detail: [{ loc: ['body'] }] }).message).toBe(
      `API 500 ${URL}`,
    );
  });

  it('honors an explicit message override', () => {
    const e = new ApiError(404, URL, { detail: 'nope' }, 'custom');
    expect(e.message).toBe('custom');
    expect(e.detail).toBe('nope');
  });
});
