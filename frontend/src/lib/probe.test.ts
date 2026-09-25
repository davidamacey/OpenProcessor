/**
 * #36 item 8 — the /train "Run probe predictions" control's pure logic.
 */
import { describe, expect, it } from 'vitest';
import { canRunProbe, classifyProbePoll } from './probe';
import type { TrainJobStatus } from './types_train';

describe('classifyProbePoll', () => {
  it('running stays running', () => {
    expect(classifyProbePoll('running')).toBe('running');
  });

  it('failed and cancelled are terminal, distinct outcomes', () => {
    expect(classifyProbePoll('failed')).toBe('failed');
    expect(classifyProbePoll('cancelled')).toBe('cancelled');
  });

  it('completed and idle both stop polling with no error (completed)', () => {
    expect(classifyProbePoll('completed')).toBe('completed');
    expect(classifyProbePoll('idle')).toBe('completed');
  });

  it('an unknown status also resolves as completed rather than polling forever', () => {
    expect(classifyProbePoll('some_future_status')).toBe('completed');
  });
});

function run(over: Partial<TrainJobStatus> = {}): TrainJobStatus {
  return {
    job_id: 'job-1',
    state: 'finished',
    checkpoint_sha256: 'deadbeef',
    ...over,
  } as TrainJobStatus;
}

describe('canRunProbe', () => {
  it('true for a finished run with an exported checkpoint', () => {
    expect(canRunProbe(run())).toBe(true);
  });

  it('false for a run that is not finished (running, failed, etc.)', () => {
    expect(canRunProbe(run({ state: 'running' }))).toBe(false);
    expect(canRunProbe(run({ state: 'failed' }))).toBe(false);
  });

  it('false for a finished run with no recorded checkpoint', () => {
    expect(canRunProbe(run({ checkpoint_sha256: null }))).toBe(false);
    expect(canRunProbe(run({ checkpoint_sha256: undefined }))).toBe(false);
  });
});
