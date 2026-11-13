import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

vi.mock('$lib/api_train', async () => {
  const actual = await vi.importActual<typeof import('$lib/api_train')>('$lib/api_train');
  return { ...actual, getTrainRuns: vi.fn(), getProbeStatus: vi.fn(), runProbe: vi.fn() };
});

import { getProbeStatus, getTrainRuns } from '$lib/api_train';
import FindDisagreementsDialog from './FindDisagreementsDialog.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;
const tick = () => new Promise((r) => setTimeout(r, 0));

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
  vi.mocked(getProbeStatus).mockResolvedValue({ status: 'idle' } as never);
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target.remove();
  vi.clearAllMocks();
});

describe('FindDisagreementsDialog', () => {
  it('offers the step under its one name for the newest probe-able run', async () => {
    vi.mocked(getTrainRuns).mockResolvedValue({
      items: [
        { job_id: 'j1', state: 'finished', checkpoint_path: '/x/best.pt' },
      ] as never,
      total: 1,
    });
    instance = mount(FindDisagreementsDialog, { target, props: { onclose: vi.fn() } });
    await tick();
    flushSync();
    const dialog = document.querySelector('[role="dialog"]')!;
    expect(dialog.textContent).toContain('Find disagreements with the new model');
    expect(document.querySelector('[data-testid="probe-control"]')).not.toBeNull();
  });

  it('says to train first when no run can be probed', async () => {
    vi.mocked(getTrainRuns).mockResolvedValue({ items: [], total: 0 });
    instance = mount(FindDisagreementsDialog, { target, props: { onclose: vi.fn() } });
    await tick();
    flushSync();
    expect(
      document.querySelector('[data-testid="find-disagreements-body"]')!.textContent,
    ).toContain('Train a model first');
  });
});
