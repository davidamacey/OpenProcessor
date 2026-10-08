/**
 * Mount-based behavior test for ProbeControl (#36 item 8) — the /train
 * "Run probe predictions" control, following CropCard.test.ts's mount
 * convention. `$lib/api`'s probe wrappers are mocked so the test drives
 * confirm/start/poll/cancel deterministically without a real backend.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import type { TrainJobStatus } from '$lib/types_train';

const getProbeStatus = vi.fn();
const runProbe = vi.fn();
const cancelProbe = vi.fn();

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getProbeStatus, runProbe, cancelProbe };
});

const { default: ProbeControl } = await import('./ProbeControl.svelte');

function status(over: Partial<TrainJobStatus> = {}): TrainJobStatus {
  return {
    job_id: 'train-1',
    state: 'finished',
    checkpoint_path: '/x/train-1/weights/best.pt',
    ...over,
  } as TrainJobStatus;
}

let target: HTMLDivElement;
let instance: unknown;

async function renderControl(props: Record<string, unknown>) {
  getProbeStatus.mockResolvedValue({ status: 'idle' });
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ProbeControl, { target, props } as never);
  flushSync();
  await Promise.resolve();
  await Promise.resolve();
  flushSync();
  return target;
}

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  vi.clearAllMocks();
});

describe('ProbeControl', () => {
  it('offers "Run probe predictions" for a finished run with an exported checkpoint', async () => {
    const el = await renderControl({ status: status() });
    const btn = el.querySelector('button');
    expect(btn?.textContent?.trim()).toBe('Run probe predictions');
    expect(btn?.disabled).toBe(false);
  });

  it('is absent for a run with no exported checkpoint and no prior result', async () => {
    const el = await renderControl({ status: status({ checkpoint_path: null }) });
    expect(el.querySelector('[data-testid="probe-control"]')).toBeNull();
  });

  it('does not poll /probe/status at all for a run that cannot qualify (no checkpoint)', async () => {
    await renderControl({ status: status({ checkpoint_path: null }) });
    expect(getProbeStatus).not.toHaveBeenCalled();
  });

  it("confirm dialog then Confirm calls runProbe with this run's job_id", async () => {
    runProbe.mockResolvedValue({
      status: 'running',
      job_id: 'p1',
      train_job_id: 'train-1',
    });
    const el = await renderControl({ status: status() });

    el.querySelector('button')?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    flushSync();
    const dialog = document.querySelector('[role="dialog"]');
    expect(dialog).toBeTruthy();

    const confirmBtn = Array.from(dialog!.querySelectorAll('button')).find(
      (b) => b.textContent?.trim() === 'Confirm',
    );
    confirmBtn?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    flushSync();
    await Promise.resolve();
    await Promise.resolve();
    flushSync();

    expect(runProbe).toHaveBeenCalledWith('train-1');
    expect(el.textContent).toContain('Running…');
  });

  it('shows the served error verbatim when the run fails', async () => {
    runProbe.mockRejectedValue(new Error('no exported checkpoint'));
    const el = await renderControl({ status: status() });

    el.querySelector('button')?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    flushSync();
    const confirmBtn = Array.from(
      document.querySelectorAll('[role="dialog"] button'),
    ).find((b) => b.textContent?.trim() === 'Confirm');
    confirmBtn?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    flushSync();
    await Promise.resolve();
    await Promise.resolve();
    flushSync();

    expect(el.textContent).toContain('Failed: no exported checkpoint');
  });

  it('adopts an in-flight probe job for this run on mount', async () => {
    getProbeStatus.mockResolvedValue({
      status: 'running',
      job_id: 'p1',
      train_job_id: 'train-1',
    });
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ProbeControl, { target, props: { status: status() } } as never);
    flushSync();
    await Promise.resolve();
    await Promise.resolve();
    flushSync();

    expect(target.textContent).toContain('Running…');
    expect(target.querySelector('button')?.textContent?.trim()).toBe('Cancel');
  });

  it('disables and explains when a probe is running for a DIFFERENT train run', async () => {
    getProbeStatus.mockResolvedValue({
      status: 'running',
      job_id: 'p1',
      train_job_id: 'train-OTHER',
    });
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ProbeControl, { target, props: { status: status() } } as never);
    flushSync();
    await Promise.resolve();
    await Promise.resolve();
    flushSync();

    expect(target.textContent).toContain('already running for another run');
    expect(target.querySelector('button')).toBeNull();
  });
});
