/**
 * BA-3 (OpenProcessor #36, c5c606f): the clustering gate reads the
 * server-computed `drained` verdict, not `total_unfinished === 0`.
 * `AutoLabelPanel` (the thing `ClusteringHandoff` wraps) makes its own
 * network calls on mount, so these tests only assert on the gate it's
 * handed — reading `AutoLabelPanel`'s own rendered gate-blocked copy
 * (it renders `gate.reason` verbatim when `gate.blocked`).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ClusteringHandoff from './ClusteringHandoff.svelte';
import type { RegionDrain } from '$lib/types';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return {
    ...actual,
    getAutoLabelStatus: vi.fn(async () => ({
      status: 'idle',
      stage: '',
      progress: null,
    })),
  };
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

function drain(overrides: Partial<RegionDrain> = {}): RegionDrain {
  return {
    pending_detection: 0,
    pending_verification: 0,
    total_unfinished: 0,
    drained: true,
    stable_for_s: 40,
    observed_at: '2026-09-25T00:00:00Z',
    ...overrides,
  };
}

describe('ClusteringHandoff gate', () => {
  it('blocks when total_unfinished is 0 but the server has not yet confirmed drained', () => {
    instance = mount(ClusteringHandoff, {
      target,
      props: {
        runState: 'idle',
        drain: drain({ drained: false, stable_for_s: 1 }),
        drainError: false,
        drainObservedAt: Date.now(),
      },
    });
    flushSync();
    expect(target.textContent).toContain('waiting for the served stability verdict');
  });

  it('is not blocked by the drain once the server reports drained: true', () => {
    instance = mount(ClusteringHandoff, {
      target,
      props: {
        runState: 'idle',
        drain: drain({ drained: true, stable_for_s: 30 }),
        drainError: false,
        drainObservedAt: Date.now(),
      },
    });
    flushSync();
    expect(target.textContent).not.toContain('items queued');
    expect(target.textContent).not.toContain('stability verdict');
    expect(target.textContent).toContain('Worklog drained as of');
  });

  it('blocks with the served count when total_unfinished > 0', () => {
    instance = mount(ClusteringHandoff, {
      target,
      props: {
        runState: 'idle',
        drain: drain({ drained: false, total_unfinished: 12 }),
        drainError: false,
        drainObservedAt: Date.now(),
      },
    });
    flushSync();
    expect(target.textContent).toContain('12 items queued');
  });

  it('blocks while an upload run is active regardless of the drain', () => {
    instance = mount(ClusteringHandoff, {
      target,
      props: {
        runState: 'uploading',
        drain: drain({ drained: true }),
        drainError: false,
        drainObservedAt: Date.now(),
      },
    });
    flushSync();
    expect(target.textContent).toContain('Finish or cancel the upload first');
  });
});
