import { describe, expect, it } from 'vitest';
import type { JobView } from '$lib/types_jobs';
import { opStatusLine } from './opStatusLine';

const job = (over: Partial<JobView>): JobView =>
  ({ id: 'j', status: 'queued', progress: {}, ...over }) as JobView;

describe('opStatusLine', () => {
  it('says nothing before a job exists', () => {
    expect(opStatusLine(null, 'Labeling')).toBeNull();
  });

  it('a queued job names the jobs ahead of it', () => {
    expect(opStatusLine(job({ queue_ahead: 2 }), 'Labeling')).toBe('Queued, 2 ahead');
    expect(opStatusLine(job({ queue_ahead: 0 }), 'Labeling')).toBe('Queued, next up');
    expect(opStatusLine(job({ queue_ahead: null }), 'Labeling')).toBe('Queued');
    expect(opStatusLine(job({}), 'Labeling')).toBe('Queued');
  });

  it('a running job reports its served progress', () => {
    expect(
      opStatusLine(
        job({ status: 'running', progress: { done: 120, total: 412 } }),
        'Labeling',
      ),
    ).toBe('Labeling 120 of 412');
    expect(opStatusLine(job({ status: 'running' }), 'Moving')).toBe('Moving\u2026');
  });

  it('shows the other live states', () => {
    expect(opStatusLine(job({ status: 'cancelling' }), 'Labeling')).toBe('Cancelling');
    expect(opStatusLine(job({ status: 'paused' }), 'Labeling')).toBe('Paused');
    expect(opStatusLine(job({ status: 'completed' }), 'Labeling')).toBeNull();
  });
});
