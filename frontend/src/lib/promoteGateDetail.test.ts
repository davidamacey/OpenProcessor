import { describe, expect, it } from 'vitest';
import { ApiError } from '$lib/api_errors';
import { promoteGateDetail } from '$lib/promote';

const err = (status: number, body: unknown) => new ApiError(status, '/x', body);

describe('promoteGateDetail', () => {
  it('reads a served gate refusal with failures, thresholds and override', () => {
    const d = promoteGateDetail(
      err(422, {
        detail: {
          message: 'gate failed',
          failures: [
            { code: 'map', message: 'mAP low', class_name: 'car' },
            { code: 'n', message: 'few samples' },
            { code: 5, message: 'bad' },
            null,
          ],
          force_allowed: true,
          override: 'force',
          thresholds: { map: 0.5 },
        },
      }),
    );
    expect(d).toEqual({
      message: 'gate failed',
      failures: [
        { code: 'map', message: 'mAP low', class_name: 'car' },
        { code: 'n', message: 'few samples', class_name: null },
      ],
      force_allowed: true,
      override: 'force',
      thresholds: { map: 0.5 },
    });
  });

  it('defaults force_allowed, override and thresholds when not served', () => {
    expect(
      promoteGateDetail(err(422, { detail: { message: 'm', failures: [] } })),
    ).toEqual({
      message: 'm',
      failures: [],
      force_allowed: false,
      override: null,
      thresholds: null,
    });
  });

  it.each([
    ['a non-422 error', err(500, { detail: { message: 'm', failures: [] } })],
    ['a non-ApiError', new Error('x')],
    ['an array detail', err(422, { detail: [] })],
    ['a missing failures list', err(422, { detail: { message: 'm' } })],
    ['a missing message', err(422, { detail: { failures: [] } })],
  ])('returns null for %s', (_n, e) => {
    expect(promoteGateDetail(e)).toBeNull();
  });
});
