import { describe, expect, it } from 'vitest';
import { JOB_STATUSES } from '$lib/types_jobs';
import { isLatestJobStatus, jobPhase } from './jobPhase';

describe('jobPhase', () => {
  it('keeps polling while the job is queued, running, paused or cancelling', () => {
    for (const s of ['queued', 'running', 'paused', 'cancelling']) {
      expect(jobPhase(s)).toBe('active');
    }
  });

  it('maps every served status to a phase, and an unknown one to failed', () => {
    expect(jobPhase('completed')).toBe('completed');
    expect(jobPhase('completed_with_errors')).toBe('completed');
    expect(jobPhase('cancelled')).toBe('cancelled');
    expect(jobPhase('failed')).toBe('failed');
    expect(jobPhase('interrupted')).toBe('failed');
    expect(jobPhase('idle')).toBe('idle');
    expect(jobPhase('something_new')).toBe('failed');
  });

  it('recognises idle and every job status, nothing else', () => {
    for (const s of ['idle', ...JOB_STATUSES]) expect(isLatestJobStatus(s)).toBe(true);
    expect(isLatestJobStatus('done')).toBe(false);
    expect(isLatestJobStatus(null)).toBe(false);
  });
});
