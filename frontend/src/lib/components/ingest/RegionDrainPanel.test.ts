import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import RegionDrainPanel from './RegionDrainPanel.svelte';
import { ApiError, getRegionDrain } from '$lib/api';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getRegionDrain: vi.fn() };
});

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target.remove();
  vi.restoreAllMocks();
});

describe('RegionDrainPanel', () => {
  it('renders the served counts', async () => {
    vi.mocked(getRegionDrain).mockResolvedValue({
      pending_detection: 5,
      pending_verification: 2,
      total_unfinished: 7,
      drained: false,
      stable_for_s: 0,
      observed_at: '2026-09-25T00:00:00Z',
    });
    instance = mount(RegionDrainPanel, { target, props: {} });
    await vi.waitFor(() => {
      flushSync();
      expect(target.textContent).toContain('7');
    });
    expect(target.textContent).toContain('5');
    expect(target.textContent).toContain('2');
  });

  it('V-1: renders the served stall_reason verbatim and the not-ready dependencies', async () => {
    vi.mocked(getRegionDrain).mockResolvedValue({
      pending_detection: 3516,
      pending_verification: 0,
      total_unfinished: 3516,
      drained: false,
      stable_for_s: 0,
      observed_at: '2026-09-25T00:00:00Z',
      stall_reason: 'segmenter unavailable since 2026-09-25T09:57:00Z',
      region_dependencies: [
        {
          role: 'detector',
          model: 'det_a',
          ready: true,
          detail: 'ready',
          unavailable_since: null,
        },
        {
          role: 'segmenter',
          model: 'seg_b',
          ready: false,
          detail: 'not loaded',
          unavailable_since: '2026-09-25T09:57:00Z',
        },
      ],
    });
    instance = mount(RegionDrainPanel, { target, props: {} });
    await vi.waitFor(() => {
      flushSync();
      expect(
        target.querySelector('[data-testid="region-drain-stall-reason"]')?.textContent,
      ).toContain('segmenter unavailable since 2026-09-25T09:57:00Z');
    });
    const deps = target.querySelector('[data-testid="region-drain-dependencies"]');
    expect(deps?.textContent).toContain('seg_b');
    expect(deps?.textContent).not.toContain('det_a');
  });

  it('renders no stall line when stall_reason is null', async () => {
    vi.mocked(getRegionDrain).mockResolvedValue({
      pending_detection: 0,
      pending_verification: 0,
      total_unfinished: 0,
      drained: true,
      stable_for_s: 5,
      observed_at: '2026-09-25T00:00:00Z',
      stall_reason: null,
      region_dependencies: [],
    });
    instance = mount(RegionDrainPanel, { target, props: {} });
    await vi.waitFor(() => {
      flushSync();
      expect(target.textContent).toContain('Drained');
    });
    expect(target.querySelector('[data-testid="region-drain-stall-reason"]')).toBeNull();
  });

  it('renders the BA-3 drained verdict and stable_for_s, not a client heuristic', async () => {
    vi.mocked(getRegionDrain).mockResolvedValue({
      pending_detection: 0,
      pending_verification: 0,
      total_unfinished: 0,
      drained: true,
      stable_for_s: 42,
      observed_at: '2026-09-25T00:00:00Z',
    });
    instance = mount(RegionDrainPanel, { target, props: {} });
    await vi.waitFor(() => {
      flushSync();
      expect(
        target.querySelector('[data-testid="region-drain-drained"]')?.textContent,
      ).toBe('yes');
    });
    expect(target.textContent).toContain('stable for 42s');
  });

  it('renders "no" for drained while total_unfinished has just reached zero (still within the stable_polls window)', async () => {
    vi.mocked(getRegionDrain).mockResolvedValue({
      pending_detection: 0,
      pending_verification: 0,
      total_unfinished: 0,
      drained: false,
      stable_for_s: 2,
      observed_at: '2026-09-25T00:00:00Z',
    });
    instance = mount(RegionDrainPanel, { target, props: {} });
    await vi.waitFor(() => {
      flushSync();
      expect(
        target.querySelector('[data-testid="region-drain-drained"]')?.textContent,
      ).toBe('no');
    });
  });

  it('shows the served detail on a 503 and never renders 0', async () => {
    vi.mocked(getRegionDrain).mockRejectedValue(
      new ApiError(503, 'x', { detail: 'opensearch outage' }),
    );
    instance = mount(RegionDrainPanel, { target, props: {} });
    await vi.waitFor(() => {
      flushSync();
      expect(target.textContent).toContain('opensearch outage');
    });
    expect(target.textContent).not.toContain('>0<');
  });

  it('calls onUpdate with the served drain and an observed-at timestamp', async () => {
    const drain = {
      pending_detection: 0,
      pending_verification: 0,
      total_unfinished: 0,
      drained: true,
      stable_for_s: 30,
      observed_at: '2026-09-25T00:00:00Z',
    };
    vi.mocked(getRegionDrain).mockResolvedValue(drain);
    const onUpdate = vi.fn();
    instance = mount(RegionDrainPanel, { target, props: { onUpdate } });
    await vi.waitFor(() => {
      expect(onUpdate).toHaveBeenCalledWith(drain, expect.any(Number));
    });
  });
});
