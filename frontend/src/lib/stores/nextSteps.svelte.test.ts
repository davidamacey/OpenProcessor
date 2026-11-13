import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ getNextSteps: vi.fn() }));
vi.mock('$lib/api_workflow', () => api);

import { nextStepsStore } from './nextSteps.svelte';

const STEP = {
  id: 'cluster',
  label: 'Cluster the items',
  detail: '120 items, none clustered',
  done: false,
  section: 'workspace',
  query: 'view=clusters',
} as const;

beforeEach(() => {
  api.getNextSteps.mockReset();
  nextStepsStore.reset();
});

describe('nextStepsStore', () => {
  it('holds the served steps, the next one and the default view', async () => {
    api.getNextSteps.mockResolvedValue({
      steps: [STEP],
      next: STEP,
      workspace_default_view: 'clusters',
    });
    await nextStepsStore.init();
    expect(nextStepsStore.next).toEqual(STEP);
    expect(nextStepsStore.defaultView).toBe('clusters');
    expect(nextStepsStore.loaded).toBe(true);
  });

  it('reads once', async () => {
    api.getNextSteps.mockResolvedValue({
      steps: [],
      next: null,
      workspace_default_view: 'gallery',
    });
    await Promise.all([nextStepsStore.init(), nextStepsStore.init()]);
    await nextStepsStore.init();
    expect(api.getNextSteps).toHaveBeenCalledTimes(1);
  });

  it('a failed read leaves no default view, so the Workspace falls back', async () => {
    api.getNextSteps.mockRejectedValue(new Error('down'));
    await nextStepsStore.init();
    expect(nextStepsStore.defaultView).toBeNull();
    expect(nextStepsStore.loaded).toBe(true);
  });

  it('a project switch drops the previous project steps', async () => {
    api.getNextSteps.mockResolvedValue({
      steps: [STEP],
      next: STEP,
      workspace_default_view: 'clusters',
    });
    await nextStepsStore.init();
    nextStepsStore.reset();
    expect(nextStepsStore.steps).toEqual([]);
    expect(nextStepsStore.loaded).toBe(false);
  });

  it('entering the Workspace re-reads, and the default view is stale until that read lands', async () => {
    api.getNextSteps.mockResolvedValueOnce({
      steps: [],
      next: null,
      workspace_default_view: 'gallery',
    });
    await nextStepsStore.init();
    expect(nextStepsStore.defaultView).toBe('gallery');

    api.getNextSteps.mockResolvedValueOnce({
      steps: [STEP],
      next: STEP,
      workspace_default_view: 'clusters',
    });
    const read = nextStepsStore.enter();
    expect(nextStepsStore.stale).toBe(true);
    await read;
    expect(nextStepsStore.stale).toBe(false);
    expect(nextStepsStore.defaultView).toBe('clusters');
  });

  it('a failed entry read is not stale for ever', async () => {
    api.getNextSteps.mockRejectedValue(new Error('down'));
    await nextStepsStore.enter();
    expect(nextStepsStore.stale).toBe(false);
    expect(nextStepsStore.defaultView).toBeNull();
  });
});
