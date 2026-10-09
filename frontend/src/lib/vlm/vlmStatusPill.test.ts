/**
 * A VLM row's served status on `/models`: the endpoint statuses name
 * themselves through the served labels (raw when unlabeled, neutral tone),
 * the Triton statuses keep their existing pill.
 */
import { describe, expect, it } from 'vitest';
import { vlmRowStatusPill } from './vlmStatusPill';
import { modelStatusPill } from '$lib/modelStatus';

const LABELS = {
  unprobed: 'Not probed yet',
  ready: 'Ready now',
  probe_failed: 'Probe failed',
};

describe('vlmRowStatusPill', () => {
  it('an endpoint status is the served label with a neutral tone', () => {
    const p = vlmRowStatusPill({ status: 'unprobed' as never, optional: false }, LABELS);
    expect(p.label).toBe('Not probed yet');
    expect(p.className).toBe(
      modelStatusPill({ status: 'not_configured', optional: false }).className,
    );
  });

  it('an endpoint status with no served label prints raw', () => {
    expect(
      vlmRowStatusPill({ status: 'unreachable' as never, optional: false }, LABELS).label,
    ).toBe('unreachable');
    expect(
      vlmRowStatusPill({ status: 'unprobed' as never, optional: false }, null).label,
    ).toBe('unprobed');
  });

  it('a Triton status keeps its pill; the served label only renames it', () => {
    const triton = modelStatusPill({ status: 'ready', optional: false });
    expect(vlmRowStatusPill({ status: 'ready', optional: false }, null)).toEqual(triton);
    expect(vlmRowStatusPill({ status: 'ready', optional: false }, LABELS)).toEqual({
      ...triton,
      label: 'Ready now',
    });
  });
});
