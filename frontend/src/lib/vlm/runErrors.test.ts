/**
 * The served words for a refused per-run VLM call: `unknown_vlm` names the
 * requested id and the valid ones, the structured refusals show their own
 * message, a pairing `validation_failed` adds the report's issue count, and
 * anything else is the error's own message.
 */
import { describe, expect, it } from 'vitest';
import { ApiError } from '$lib/api';
import { vlmRunErrorText } from './runErrors';

const err = (status: number, detail: unknown) => new ApiError(status, '/x', { detail });

describe('vlmRunErrorText', () => {
  it('unknown_vlm: the requested id and the valid ids', () => {
    expect(
      vlmRunErrorText(
        err(422, {
          error: 'unknown_vlm',
          message: 'Unknown VLM.',
          axis: 'vlm',
          requested: 'nope',
          valid_ids: ['local_vlm', 'off'],
        }),
      ),
    ).toBe('unknown vlm "nope" — valid: local_vlm, off.');
  });

  it.each([
    [422, 'vlm_external_not_acknowledged', 'Acknowledge cloud_vlm first.'],
    [409, 'vlm_not_configured', 'No VLM is configured for this project.'],
    [409, 'vlm_endpoint_unavailable', 'cloud_vlm is unreachable.'],
  ])('%s %s shows the served message verbatim', (status, error, message) => {
    expect(vlmRunErrorText(err(status, { error, message }))).toBe(message);
  });

  it('a pairing validation_failed adds the served issue count', () => {
    const issue = (code: string) => ({
      code,
      severity: 'error',
      field: null,
      message: 'm',
      detail: {},
      bypassable: false,
    });
    expect(
      vlmRunErrorText(
        err(422, {
          error: 'validation_failed',
          message: 'The pack does not fit this endpoint.',
          report: {
            ok: false,
            errors: [issue('a'), issue('b')],
            warnings: [],
            force_allowed: false,
          },
        }),
      ),
    ).toBe('The pack does not fit this endpoint. (2 issues)');
  });

  it('anything else is the error message', () => {
    expect(vlmRunErrorText(new Error('boom'))).toBe('boom');
  });
});
