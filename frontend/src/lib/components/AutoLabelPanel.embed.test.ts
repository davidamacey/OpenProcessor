/**
 * v0.4.0 `embed_missing` on the auto-label run: the checkbox sends
 * `embed_missing=true` only when ticked; the served stage has a label; a
 * failed job shows its served error and the stage table marks an errored
 * stage.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import AutoLabelPanel from './AutoLabelPanel.svelte';
import { getAutoLabelStatus, startAutoLabel } from '$lib/api';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return {
    ...actual,
    getAutoLabelStatus: vi.fn(),
    startAutoLabel: vi.fn(),
  };
});

const IDLE = { status: 'idle', stage: '', progress: null };

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
  vi.mocked(getAutoLabelStatus).mockResolvedValue(IDLE as never);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

const q = (id: string) => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);
const button = (text: string) =>
  [...document.querySelectorAll('button')].find((b) => b.textContent?.trim() === text);

async function start(tick: boolean) {
  vi.mocked(startAutoLabel).mockResolvedValue({ ...IDLE, status: 'running' } as never);
  instance = mount(AutoLabelPanel, { target, props: {} });
  flushSync();
  if (tick) {
    const box = q('embed-missing-checkbox') as HTMLInputElement;
    box.checked = true;
    box.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
  }
  button('Recluster now')!.click();
  flushSync();
  button('Start')!.click();
  await vi.waitFor(() => expect(startAutoLabel).toHaveBeenCalledTimes(1));
  return vi.mocked(startAutoLabel).mock.calls[0]![0] as Record<string, unknown>;
}

describe('AutoLabelPanel embed_missing', () => {
  it('omits embed_missing when the box is not ticked', async () => {
    const params = await start(false);
    expect(params).not.toHaveProperty('embed_missing');
  });

  it('sends embed_missing=true when ticked', async () => {
    const params = await start(true);
    expect(params.embed_missing).toBe(true);
  });

  it('names the embed_missing stage and counts it first when the run asked for it', async () => {
    vi.mocked(getAutoLabelStatus).mockResolvedValue({
      job_id: 'j',
      status: 'running',
      stage: 'embed_missing',
      processed: 0,
      total: 0,
      args: { embed_missing: true },
      result: {},
    } as never);
    instance = mount(AutoLabelPanel, { target, props: {} });
    await vi.waitFor(() => {
      flushSync();
      expect(target.textContent).toContain('embedding items without a vector');
    });
    expect(target.textContent).toContain('stage 1/6');
  });

  it('a failed job shows the served error and marks the errored stage', async () => {
    vi.mocked(getAutoLabelStatus).mockResolvedValue({
      job_id: 'j',
      status: 'failed',
      stage: 'embed_missing',
      processed: 0,
      total: 0,
      error: 'embedding service unavailable',
      finished_at: 1,
      args: { embed_missing: true },
      result: {
        stages: {
          embed_missing: { status: 'error', reason: 'encoder down' },
        },
      },
    } as never);
    instance = mount(AutoLabelPanel, { target, props: {} });
    await vi.waitFor(() => {
      flushSync();
      expect(target.textContent).toContain('Error: embedding service unavailable');
    });
    const row = target.querySelector(
      '[data-testid="last-run-stages"] tr[data-stage-status]',
    )!;
    expect(row.getAttribute('data-stage-status')).toBe('error');
    expect(row.className).toContain('red');
    expect(row.textContent).toContain('embedding items without a vector');
  });
});
