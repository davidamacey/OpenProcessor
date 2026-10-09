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
    checkpoint_path: '/var/lib/openprocessor/training_runs/job-1/weights/best.pt',
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
    expect(canRunProbe(run({ checkpoint_path: null }))).toBe(false);
    expect(canRunProbe(run({ checkpoint_path: undefined }))).toBe(false);
  });

  // Live regression (2026-09-25): GET {API_PREFIX}/train/status/{job_id}
  // serves checkpoint_path on a finished run but NOT checkpoint_sha256
  // (that only appears inside the manifest) — gating on the sha would
  // hide the button for every real finished run on this deployment.
  it('true even when checkpoint_sha256 is absent, as long as checkpoint_path is served', () => {
    expect(
      canRunProbe(run({ checkpoint_sha256: undefined, checkpoint_path: '/x/best.pt' })),
    ).toBe(true);
  });
});
