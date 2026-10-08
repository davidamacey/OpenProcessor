/**
 * The job polls of ScoresCard, ProbeControl and EmbeddingPlot are a
 * setInterval plus an async status read. Two races: (A) the component
 * unmounts while the first "adopt an in-flight job" read is pending, and the
 * read then starts an interval nothing will ever stop; (B) a status read
 * slower than the interval overlaps the next one and both run the completion.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

const getScoresCoverage = vi.fn();
const getScoresStatus = vi.fn();
const getMethods = vi.fn();
const getProbeStatus = vi.fn();
const getVizProjection = vi.fn();
const getVizProjectionStatus = vi.fn();

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return {
    ...actual,
    getScoresCoverage: (...a: unknown[]) => getScoresCoverage(...a),
    getScoresStatus: (...a: unknown[]) => getScoresStatus(...a),
    getMethods: (...a: unknown[]) => getMethods(...a),
    getProbeStatus: (...a: unknown[]) => getProbeStatus(...a),
    getVizProjection: (...a: unknown[]) => getVizProjection(...a),
    getVizProjectionStatus: (...a: unknown[]) => getVizProjectionStatus(...a),
  };
});

const { toastStore } = await import('$lib/stores/toast.svelte');
const { default: ScoresCard } = await import('./ScoresCard.svelte');
const { default: ProbeControl } = await import('./ProbeControl.svelte');
const { default: EmbeddingPlot } = await import('./EmbeddingPlot.svelte');

const scoresJob = (status: string) => ({
  job_id: 'j',
  status,
  scorers: [],
  processed: 0,
  total: 0,
  started_at: 0,
  finished_at: 0,
  error: null,
  results: {},
});
const probeJob = (status: string) => ({
  status,
  job_id: 'p',
  train_job_id: 'train-1',
  updated_count: 5,
});
const vizJob = (status: string) => ({ status, job_id: 'v', n_written: 9, error: null });

const PROBE_PROPS = {
  status: { job_id: 'train-1', state: 'finished', checkpoint_path: '/x/best.pt' },
};

let target: HTMLDivElement;

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe(): void {}
      unobserve(): void {}
      disconnect(): void {}
    },
  );
  for (const m of [
    getScoresCoverage,
    getScoresStatus,
    getMethods,
    getProbeStatus,
    getVizProjection,
    getVizProjectionStatus,
  ])
    m.mockReset();
  getScoresCoverage.mockResolvedValue({});
  getMethods.mockResolvedValue({});
  getVizProjection.mockResolvedValue({ built: true, points: [] });
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  target.remove();
  vi.restoreAllMocks();
});

type Case = {
  name: string;
  status: ReturnType<typeof vi.fn>;
  running: unknown;
  mountIt: () => ReturnType<typeof mount>;
};

const CASES: Case[] = [
  {
    name: 'ScoresCard',
    status: getScoresStatus,
    running: scoresJob('running'),
    mountIt: () => mount(ScoresCard, { target }),
  },
  {
    name: 'ProbeControl',
    status: getProbeStatus,
    running: probeJob('running'),
    mountIt: () => mount(ProbeControl, { target, props: PROBE_PROPS } as never),
  },
  {
    name: 'EmbeddingPlot',
    status: getVizProjectionStatus,
    running: vizJob('running'),
    mountIt: () => mount(EmbeddingPlot, { target }),
  },
];

describe.each(CASES)('$name job poll', ({ status, running, mountIt }) => {
  it('starts no poll when the component unmounts before the adopt read resolves', async () => {
    let release!: (v: unknown) => void;
    status.mockImplementationOnce(() => new Promise((r) => (release = r)));
    status.mockResolvedValue(running);
    const inst = mountIt();
    flushSync();
    await vi.advanceTimersByTimeAsync(10);
    const before = status.mock.calls.length;
    expect(before).toBe(1);
    unmount(inst);
    release(running);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(status.mock.calls.length).toBe(before);
  });

  it('control: an adopted running job keeps polling while mounted', async () => {
    status.mockResolvedValue(running);
    const inst = mountIt();
    flushSync();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(status.mock.calls.length).toBeGreaterThan(2);
    unmount(inst);
  });
});

describe('overlapping slow status reads', () => {
  it('ScoresCard runs the completion once when a read takes longer than the interval', async () => {
    const toasts = vi.spyOn(toastStore, 'success');
    getScoresStatus.mockResolvedValueOnce(scoresJob('running'));
    let inFlight = 0;
    let maxInFlight = 0;
    getScoresStatus.mockImplementation(() => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      return new Promise((r) =>
        setTimeout(() => {
          inFlight--;
          r(scoresJob('completed'));
        }, 5000),
      );
    });
    const inst = mount(ScoresCard, { target });
    flushSync();
    await vi.advanceTimersByTimeAsync(20_000);
    expect(maxInFlight).toBe(1);
    expect(
      toasts.mock.calls.filter((c) => String(c[0]).includes('computed')),
    ).toHaveLength(1);
    unmount(inst);
  });

  it('ProbeControl runs the completion once when a read takes longer than the interval', async () => {
    const toasts = vi.spyOn(toastStore, 'success');
    getProbeStatus.mockResolvedValueOnce(probeJob('running'));
    getProbeStatus.mockImplementation(
      () => new Promise((r) => setTimeout(() => r(probeJob('completed')), 5000)),
    );
    const inst = mount(ProbeControl, { target, props: PROBE_PROPS } as never);
    flushSync();
    await vi.advanceTimersByTimeAsync(20_000);
    expect(
      toasts.mock.calls.filter((c) => String(c[0]).includes('Probe finished')),
    ).toHaveLength(1);
    unmount(inst);
  });
});
