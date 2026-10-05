/** Promote is a background job (OpenProcessor #87): 202 PromoteJobStatus. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, getPromoteJob, promoteTrainJob } from './api';

const JOB = { promote_id: 'p1', job_id: 'run 1', triton_name: 'm', status: 'queued' };

function stub(status: number, body: unknown) {
  const f = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), {
      status,
      headers: { 'content-type': 'application/json' },
    }),
  );
  vi.stubGlobal('fetch', f);
  return f;
}

afterEach(() => vi.unstubAllGlobals());

describe('promote job api', () => {
  it('POSTs without wait=true and returns the 202 job as served', async () => {
    const f = stub(202, JOB);
    const res = await promoteTrainJob('run 1', { triton_name: 'm' });
    expect(f.mock.calls[0]![0]).toBe(`${API_PREFIX}/train/promote/run%201`);
    expect(f.mock.calls[0]![1].method).toBe('POST');
    expect(res).toEqual(JOB);
  });

  it('GETs the job status route with both ids encoded', async () => {
    const f = stub(200, { ...JOB, status: 'building' });
    const res = await getPromoteJob('run 1', 'p/1');
    expect(f.mock.calls[0]![0]).toBe(`${API_PREFIX}/train/promote/run%201/jobs/p%2F1`);
    expect(res.status).toBe('building');
  });
});
