/**
 * The /train GPU picker renders the backend's allowed claims
 * (`GET /train/gpus`) instead of a hardcoded list, so a GPU the backend
 * has taken away (GPU 0 after the 2026-09-24 re-placement) is never
 * offered.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, defaultGpuValue, getTrainGpus } from './api';
import type { TrainGpuOptionsResponse } from './api';

const LIVE: TrainGpuOptionsResponse = {
  options: [
    {
      value: '2',
      gpu_ids: [2],
      label: 'RTX A6000 (GPU 2)',
      advisory: 'Stops vllm-gemma4-e4b for the run; restarted when it ends.',
      stops_containers: ['vllm-gemma4-e4b'],
      default: true,
    },
  ],
  allowed_ids: [2],
  unrestricted: false,
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('getTrainGpus', () => {
  it('GETs the served options and returns them unchanged', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(LIVE), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const res = await getTrainGpus();
    expect(fetchMock.mock.calls[0]![0]).toBe(`${API_PREFIX}/train/gpus`);
    expect(res).toEqual(LIVE);
  });
});

describe('defaultGpuValue', () => {
  it('picks the option the backend marks default, not the first one', () => {
    const res: TrainGpuOptionsResponse = {
      ...LIVE,
      options: [
        { ...LIVE.options[0]!, value: '3', gpu_ids: [3], default: false },
        LIVE.options[0]!,
      ],
      allowed_ids: [2, 3],
    };
    expect(defaultGpuValue(res)).toBe('2');
  });

  it("is '' when no option is marked default, so the claim is omitted", () => {
    const res = { ...LIVE, options: [{ ...LIVE.options[0]!, default: false }] };
    expect(defaultGpuValue(res)).toBe('');
  });
});
