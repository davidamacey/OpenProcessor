import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
  selectDiverse: vi.fn(),
  getSelectStatus: vi.fn(),
  getDiverseItems: vi.fn(),
  cancelSelect: vi.fn(),
}));
vi.mock('$lib/api_select', () => api);

import { createDiverseSample } from './diverseSample.svelte';

const FILTER = { class_names: ['widget'] };

function make() {
  const selectItems = vi.fn();
  const sample = createDiverseSample({ itemBody: () => FILTER, selectItems });
  return { sample, selectItems };
}

beforeEach(() => {
  vi.useFakeTimers();
  Object.values(api).forEach((f) => f.mockReset());
});
afterEach(() => vi.useRealTimers());

describe('diverse sample', () => {
  it('sends the Workspace filter as the scope and selects the ready result', async () => {
    api.selectDiverse.mockResolvedValue({
      kind: 'ready',
      selection: { crop_ids: ['a', 'b'], method: 'm', version: 'v', n_pool: 9 },
    });
    const { sample, selectItems } = make();
    sample.k = 2;
    await sample.start();
    expect(api.selectDiverse).toHaveBeenCalledWith({ item_filter: FILTER }, 2);
    expect(selectItems).toHaveBeenCalledWith(['a', 'b']);
    expect(sample.phase).toBe('idle');
    expect(sample.message).toBe('Picked 2 of 9');
  });

  it('polls the served job and selects its result when it completes', async () => {
    api.selectDiverse.mockResolvedValue({ kind: 'job', job_id: 'j1' });
    api.getSelectStatus
      .mockResolvedValueOnce({ job_id: 'j1', status: 'running', result: null })
      .mockResolvedValueOnce({
        job_id: 'j1',
        status: 'completed',
        result: { method: 'm', version: 'v', n_pool: 400, n_selected: 3 },
      });
    api.getDiverseItems
      .mockResolvedValueOnce({ ids: ['x', 'y'], next_cursor: 'c1' })
      .mockResolvedValueOnce({ ids: ['z'], next_cursor: null });
    const { sample, selectItems } = make();
    await sample.start();
    expect(sample.phase).toBe('running');
    await vi.advanceTimersByTimeAsync(2000);
    expect(sample.phase).toBe('running');
    await vi.advanceTimersByTimeAsync(2000);
    expect(api.getDiverseItems.mock.calls.map((c) => [c[0], c[1]])).toEqual([
      ['j1', null],
      ['j1', 'c1'],
    ]);
    expect(selectItems).toHaveBeenCalledWith(['x', 'y', 'z']);
    expect(sample.phase).toBe('idle');
    expect(sample.message).toBe('Picked 3 of 400');
  });

  it('says why when the feature is off or a selection is already running', async () => {
    api.selectDiverse.mockResolvedValueOnce({ kind: 'disabled' });
    const a = make();
    await a.sample.start();
    expect(a.sample.message).toMatch(/turned off/);
    api.selectDiverse.mockResolvedValueOnce({ kind: 'already_running' });
    const b = make();
    await b.sample.start();
    expect(b.sample.message).toMatch(/already running/);
    expect(b.selectItems).not.toHaveBeenCalled();
  });

  it('cancel stops the job and the polling', async () => {
    api.selectDiverse.mockResolvedValue({ kind: 'job', job_id: 'j1' });
    api.getSelectStatus.mockResolvedValue({ status: 'running', result: null });
    api.cancelSelect.mockResolvedValue(undefined);
    const { sample } = make();
    await sample.start();
    await sample.cancel();
    expect(api.cancelSelect).toHaveBeenCalledOnce();
    expect(sample.phase).toBe('idle');
    api.getSelectStatus.mockClear();
    await vi.advanceTimersByTimeAsync(10000);
    expect(api.getSelectStatus).not.toHaveBeenCalled();
  });

  it('a failed job reports its error and selects nothing', async () => {
    api.selectDiverse.mockResolvedValue({ kind: 'job', job_id: 'j1' });
    api.getSelectStatus.mockResolvedValue({ status: 'failed', error: 'no vectors' });
    const { sample, selectItems } = make();
    await sample.start();
    await vi.advanceTimersByTimeAsync(2000);
    expect(sample.message).toContain('no vectors');
    expect(selectItems).not.toHaveBeenCalled();
    expect(sample.phase).toBe('idle');
  });
});
