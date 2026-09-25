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
