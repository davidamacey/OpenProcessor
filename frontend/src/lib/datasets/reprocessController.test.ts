/**
 * ReprocessFlow (W10.13): a single image applies from the dialog with dry_run false;
 * the batch apply needs a served dry run for the
 * current choices; a changed scope invalidates it; `region_mode` is sent
 * only with the region scope; a served job is followed until its
 * `poll_after_s` is null, and can be cancelled.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { reprocessFixture } from '$lib/test/fixtures/datasetImport';
import { ReprocessFlow } from './reprocessController.svelte';

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

function deps() {
  return {
    reprocessBatch: vi.fn(),
    reprocessCrop: vi.fn(),
    reprocessImage: vi.fn(),
    getReprocessJob: vi.fn(),
    cancelReprocessJob: vi.fn(),
  };
}

describe('ReprocessFlow', () => {
  it('batch apply needs a dry run for the current choices', async () => {
    const d = deps();
    d.reprocessBatch.mockResolvedValue(reprocessFixture());
    const f = new ReprocessFlow({ kind: 'crops', cropIds: ['a'] }, d);
    f.toggleScope('embed', true);
    f.setRegionMode('reverify');
    expect(f.canApply).toBe(false);
    await f.preview();
    expect(d.reprocessBatch).toHaveBeenLastCalledWith({
      targets: { crop_ids: ['a'] },
      scopes: ['embed'],
      dry_run: true,
    });
    expect(f.canApply).toBe(true);
    f.toggleScope('region', true);
    expect(f.dryRun).toBeNull();
    expect(f.canApply).toBe(false);
    await f.preview();
    expect(d.reprocessBatch.mock.calls.at(-1)![0]).toMatchObject({
      scopes: ['embed', 'region'],
      region_mode: 'reverify',
    });
  });

  it('follows a served job until poll_after_s is null, and cancels it', async () => {
    const d = deps();
    d.reprocessBatch.mockResolvedValueOnce(reprocessFixture()).mockResolvedValueOnce(
      reprocessFixture({
        dry_run: false,
        job: { job_id: 'rj1', status: 'running', poll_after_s: 2 },
      }),
    );
    d.getReprocessJob
      .mockResolvedValueOnce({ job_id: 'rj1', status: 'running', poll_after_s: 2 })
      .mockResolvedValueOnce({ job_id: 'rj1', status: 'completed', poll_after_s: null });
    d.cancelReprocessJob.mockResolvedValue({
      job_id: 'rj1',
      status: 'cancelled',
      poll_after_s: null,
    });
    const f = new ReprocessFlow({ kind: 'crops', cropIds: ['a', 'b'] }, d);
    f.toggleScope('detect', true);
    await f.preview();
    await f.apply();
    expect(f.job?.status).toBe('running');
    await vi.advanceTimersByTimeAsync(2000);
    expect(d.getReprocessJob).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(2000);
    expect(f.job?.status).toBe('completed');
    await vi.advanceTimersByTimeAsync(10_000);
    expect(d.getReprocessJob).toHaveBeenCalledTimes(2);
    await f.cancelJob();
    expect(d.cancelReprocessJob).toHaveBeenCalledWith('rj1');
    expect(f.job?.status).toBe('cancelled');
  });

  it('one crop applies with dry_run false and returns the served crops', async () => {
    const d = deps();
    d.reprocessCrop.mockResolvedValue(
      reprocessFixture({ dry_run: false, items: [{ id: 'c1' }] }),
    );
    const f = new ReprocessFlow({ kind: 'crop', cropId: 'c1' }, d);
    expect(await f.apply()).toEqual([]);
    f.toggleScope('region', true);
    expect(await f.apply()).toEqual([{ id: 'c1' }]);
    expect(d.reprocessCrop).toHaveBeenCalledWith('c1', {
      scopes: ['region'],
      dry_run: false,
    });
    expect(f.canApply).toBe(false);
  });

  it('one image applies with dry_run false and hands back every served item', async () => {
    const d = deps();
    d.reprocessImage.mockResolvedValue(
      reprocessFixture({ dry_run: false, items: [{ id: 'c1' }, { id: 'c2' }] }),
    );
    const f = new ReprocessFlow({ kind: 'image', imageId: 'img_1' }, d);
    expect(f.isBatch).toBe(false);
    expect(f.count).toBe(1);
    f.toggleScope('detect', true);
    f.toggleScope('region', true);
    f.setRegionMode('reverify');
    expect(await f.apply()).toEqual([{ id: 'c1' }, { id: 'c2' }]);
    expect(d.reprocessImage).toHaveBeenCalledWith('img_1', {
      scopes: ['detect', 'region'],
      region_mode: 'reverify',
      dry_run: false,
    });
    expect(d.reprocessCrop).not.toHaveBeenCalled();
    expect(d.reprocessBatch).not.toHaveBeenCalled();
  });

  it('a served request (W4 suggested_reprocess) is sent exactly as served, dry run first', async () => {
    const d = deps();
    d.reprocessBatch.mockResolvedValue(reprocessFixture());
    const request = {
      targets: { filter: { profile_not: 'widget_tag', include_detected: true } },
      scopes: ['region' as const],
      region_mode: 'redetect' as const,
      dry_run: true,
    };
    const f = new ReprocessFlow({ kind: 'request', request }, d);
    expect(f.isBatch).toBe(true);
    expect(f.scopes).toEqual(['region']);
    f.toggleScope('embed', true);
    f.setRegionMode('reverify');
    expect(f.scopes).toEqual(['region']);
    expect(f.canApply).toBe(false);
    await f.preview();
    expect(d.reprocessBatch).toHaveBeenLastCalledWith({ ...request, dry_run: true });
    expect(f.canApply).toBe(true);
    await f.apply();
    expect(d.reprocessBatch).toHaveBeenLastCalledWith({ ...request, dry_run: false });
  });
});
