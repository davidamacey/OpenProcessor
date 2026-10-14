/**
 * The bulk-ingest controller: follows the served run by its `poll_after_s`
 * until it is null, adopts a run already live, and sends start / cancel /
 * resume as the backend allows. Nothing is decided client-side.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import {
  RUN_ID,
  configFixture,
  progressFixture,
  runFixture,
} from '$lib/test/fixtures/bulkIngest';
import { BulkIngest } from './bulkIngestController.svelte';

function make(sequence: ReturnType<typeof runFixture>[], recent = sequence.slice(0, 1)) {
  let i = 1; // sequence[0] is what start()/init() adopt; polls continue from the next
  const deps = {
    getBulkIngestConfig: vi.fn().mockResolvedValue(configFixture()),
    startBulkIngest: vi.fn().mockResolvedValue(sequence[0] ?? runFixture()),
    listBulkIngests: vi
      .fn()
      .mockResolvedValue({ items: recent, total: recent.length, page: 1, page_size: 5 }),
    getBulkIngest: vi
      .fn()
      .mockImplementation(async () => sequence[Math.min(i++, sequence.length - 1)]),
    getBulkIngestFailures: vi
      .fn()
      .mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 }),
    cancelBulkIngest: vi
      .fn()
      .mockResolvedValue(runFixture({ status: 'cancelled', poll_after_s: null })),
  };
  return { ctl: new BulkIngest(deps), deps };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('following a run', () => {
  it('polls after the served poll_after_s and stops when it is null', async () => {
    const done = runFixture({
      status: 'completed',
      poll_after_s: null,
      can_cancel: false,
      progress: progressFixture({ discovered: 5, processed: 5, ingested: 5 }),
    });
    const { ctl, deps } = make([runFixture({ poll_after_s: 3 }), runFixture(), done]);
    await ctl.start();
    expect(ctl.live).toBe(true);
    await vi.advanceTimersByTimeAsync(2999);
    expect(deps.getBulkIngest).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(deps.getBulkIngest).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(2000);
    expect(ctl.run?.status).toBe('completed');
    expect(ctl.live).toBe(false);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(deps.getBulkIngest).toHaveBeenCalledTimes(2);
    ctl.stop();
  });

  it('keeps following a live run through a failed read', async () => {
    const { ctl, deps } = make([runFixture({ poll_after_s: 2 })]);
    await ctl.start();
    deps.getBulkIngest.mockRejectedValueOnce(new Error('network'));
    await vi.advanceTimersByTimeAsync(2000);
    expect(ctl.loadError).toBe('network');
    await vi.advanceTimersByTimeAsync(2000);
    expect(ctl.loadError).toBeNull();
    expect(deps.getBulkIngest).toHaveBeenCalledTimes(2);
    ctl.stop();
  });

  it('adopts the newest run on init so a reload keeps showing progress', async () => {
    const live = runFixture({
      progress: progressFixture({ discovered: 40, processed: 10 }),
    });
    const { ctl, deps } = make([live], [live]);
    await ctl.init();
    expect(ctl.config?.default_concurrency).toBe(8);
    expect(ctl.run?.run_id).toBe(RUN_ID);
    await vi.advanceTimersByTimeAsync(2000);
    expect(deps.getBulkIngest).toHaveBeenCalledWith(RUN_ID);
    ctl.stop();
  });

  it('loads the failures once a run with failed images is over', async () => {
    const failed = runFixture({
      status: 'completed_with_errors',
      poll_after_s: null,
      can_cancel: false,
      can_resume: true,
      progress: progressFixture({ failed: 2 }),
    });
    const { ctl, deps } = make([failed]);
    deps.startBulkIngest.mockResolvedValue(failed);
    deps.getBulkIngestFailures.mockResolvedValue({
      items: [
        { path: '/data/source/a.jpg', error: 'cannot read', error_kind: 'decode_failed' },
      ],
      total: 2,
      offset: 0,
      limit: 20,
    });
    await ctl.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(deps.getBulkIngestFailures).toHaveBeenCalledWith(RUN_ID, {
      offset: 0,
      limit: 20,
    });
    expect(ctl.failures?.total).toBe(2);
  });
});

describe('actions', () => {
  it('sends only what the form states, leaving options unset for the served defaults', () => {
    const { ctl } = make([]);
    ctl.form.path = '  /data/source/coco  ';
    expect(ctl.request).toEqual({ folder: '/data/source/coco', source: 'bulk_ingest' });
    ctl.form.kind = 'manifest';
    ctl.form.path = '/data/source/list.txt';
    ctl.form.source = 'census';
    ctl.form.extensions = ' .jpg, ,.png ';
    ctl.form.batchSize = 8;
    ctl.form.concurrency = 4;
    expect(ctl.request).toEqual({
      manifest: '/data/source/list.txt',
      source: 'census',
      extensions: ['.jpg', '.png'],
      batch_size: 8,
      concurrency: 4,
    });
  });

  it('cannot start without a path or while a run is live', async () => {
    const { ctl } = make([]);
    expect(ctl.canStart).toBe(false);
    ctl.form.path = '/data/source/a';
    expect(ctl.canStart).toBe(true);
    await ctl.start();
    expect(ctl.canStart).toBe(false);
    ctl.stop();
  });

  it('shows the served refusal of a start and stays startable', async () => {
    const { ctl, deps } = make([]);
    deps.startBulkIngest.mockRejectedValue(
      new ApiError(
        409,
        '/ingest/bulk',
        {
          detail: {
            error: 'bulk_ingest_busy',
            message: 'a bulk ingest is already running',
          },
        },
        undefined,
      ),
    );
    ctl.form.path = '/data/source/a';
    expect(await ctl.start()).toBe(false);
    expect(ctl.actionError).toContain('already running');
    expect(ctl.canStart).toBe(true);
  });

  it('cancel posts for the current run and adopts the served answer', async () => {
    const { ctl, deps } = make([]);
    await ctl.start();
    expect(await ctl.cancel()).toBe(true);
    expect(deps.cancelBulkIngest).toHaveBeenCalledWith(RUN_ID);
    expect(ctl.run?.status).toBe('cancelled');
    expect(ctl.live).toBe(false);
  });

  it('resume starts the run source again with its own options', async () => {
    const stopped = runFixture({
      status: 'cancelled',
      poll_after_s: null,
      can_cancel: false,
      can_resume: true,
      options: { source_tag: 'census', batch_size: 8, concurrency: 4 },
    });
    const { ctl, deps } = make([]);
    await ctl.resume(stopped);
    expect(deps.startBulkIngest).toHaveBeenCalledWith({
      folder: '/data/source/coco',
      source: 'census',
      extensions: ['.jpg'],
      batch_size: 8,
      concurrency: 4,
    });
    ctl.stop();
  });
});
