/**
 * The import job view's controller: follows the served job by its
 * `poll_after_s` until terminal, wakes on a matching SSE event, and runs
 * cancel / resume / undo (dry run first) as served (W10.11, W10.12).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import { jobFixture, undoReportFixture } from '$lib/test/fixtures/datasetImport';
import type { CurationEvent } from '$lib/sse';
import { ImportJob } from './importJobController.svelte';

const ID = 'imp_20260927T120000_1a2b3c4d';

function make(sequence: ReturnType<typeof jobFixture>[]) {
  let i = 0;
  let onEvent: ((e: CurationEvent) => void) | null = null;
  const close = vi.fn();
  const deps = {
    getDatasetImport: vi
      .fn()
      .mockImplementation(async () => sequence[Math.min(i++, sequence.length - 1)]),
    cancelDatasetImport: vi
      .fn()
      .mockResolvedValue(jobFixture({ status: 'cancelled', poll_after_s: null })),
    resumeDatasetImport: vi.fn().mockResolvedValue(jobFixture({ status: 'running' })),
    undoDatasetImport: vi.fn(),
    runServedNextStep: vi.fn().mockResolvedValue({}),
    getDatasetImportIssues: vi
      .fn()
      .mockResolvedValue({ items: [], total: 0, page: 1, page_size: 20 }),
    getDatasetImportEntries: vi
      .fn()
      .mockResolvedValue({ items: [], total: 0, page: 1, page_size: 20 }),
    subscribe: vi.fn().mockImplementation((cb: (e: CurationEvent) => void) => {
      onEvent = cb;
      return { close };
    }),
  };
  const job = new ImportJob(ID, deps);
  return { job, deps, emit: (e: CurationEvent) => onEvent?.(e), close };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('following a job', () => {
  it('polls after the served poll_after_s until it is null (terminal)', async () => {
    const done = jobFixture({
      status: 'completed',
      poll_after_s: null,
      progress: { ...jobFixture().progress, images_done: 82 },
    });
    const { job, deps } = make([
      jobFixture({ status: 'running', poll_after_s: 2 }),
      jobFixture({ status: 'running', poll_after_s: 2 }),
      done,
    ]);
    job.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(job.status).toBe('running');
    expect(job.statusLabel()).toBe('Importing');
    await vi.advanceTimersByTimeAsync(1999);
    expect(deps.getDatasetImport).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(deps.getDatasetImport).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(2000);
    expect(job.status).toBe('completed');
    expect(job.statusLabel()).toBe('Done');
    await vi.advanceTimersByTimeAsync(10_000);
    expect(deps.getDatasetImport).toHaveBeenCalledTimes(3);
    job.stop();
  });

  it('a failed job keeps its served error and stops polling', async () => {
    const { job, deps } = make([
      jobFixture({
        status: 'failed',
        poll_after_s: null,
        error: '6 consecutive chunks failed: index unavailable.',
      }),
    ]);
    job.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(job.job?.error).toBe('6 consecutive chunks failed: index unavailable.');
    expect(job.canResume).toBe(true);
    expect(job.canCancel).toBe(false);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(deps.getDatasetImport).toHaveBeenCalledTimes(1);
  });

  it('an SSE event for this import wakes an immediate re-read; others are ignored', async () => {
    const { job, deps, emit, close } = make([jobFixture({ poll_after_s: 30 })]);
    job.start();
    await vi.advanceTimersByTimeAsync(0);
    emit({ type: 'dataset_import.progress', import_id: 'imp_other' } as CurationEvent);
    emit({ type: 'crop.created', crop_id: 'c' } as CurationEvent);
    await vi.advanceTimersByTimeAsync(0);
    expect(deps.getDatasetImport).toHaveBeenCalledTimes(1);
    emit({ type: 'dataset_import.progress', import_id: ID } as CurationEvent);
    await vi.advanceTimersByTimeAsync(0);
    expect(deps.getDatasetImport).toHaveBeenCalledTimes(2);
    job.stop();
    expect(close).toHaveBeenCalled();
  });

  it('a load failure shows the served message', async () => {
    const { job, deps } = make([]);
    deps.getDatasetImport.mockRejectedValue(
      new ApiError(404, '/x', {
        detail: { error: 'import_not_found', message: 'No such import.' },
      }),
    );
    await job.load();
    expect(job.loadError).toBe('No such import.');
  });
});

describe('actions', () => {
  it('status decides which actions exist', async () => {
    const { job } = make([
      jobFixture({ status: 'paused_backpressure', poll_after_s: null }),
    ]);
    await job.load();
    expect([job.canCancel, job.canResume, job.canUndo]).toEqual([true, false, false]);
  });

  it('cancel adopts the served job; a 409 shows its served message', async () => {
    const { job, deps } = make([jobFixture({ poll_after_s: null })]);
    await job.load();
    expect(await job.cancel()).toBe(true);
    expect(job.status).toBe('cancelled');
    deps.resumeDatasetImport.mockRejectedValue(
      new ApiError(409, '/x', {
        detail: {
          error: 'dataset_changed',
          message: 'The dataset changed since this import.',
        },
      }),
    );
    expect(await job.resume()).toBe(false);
    expect(job.actionError).toBe('The dataset changed since this import.');
  });

  it('undo: dry run returns the served report, apply sends dry_run false', async () => {
    const { job, deps } = make([jobFixture({ status: 'completed', poll_after_s: null })]);
    await job.load();
    deps.undoDatasetImport
      .mockResolvedValueOnce(undoReportFixture())
      .mockResolvedValueOnce(jobFixture({ status: 'undoing', poll_after_s: 2 }));
    const choices = { remove_images: true, deprecate_created_classes: false };
    const report = await job.undoDryRun(choices);
    expect(report?.items_kept_human_edited).toBe(2);
    expect(deps.undoDatasetImport).toHaveBeenLastCalledWith(ID, {
      dry_run: true,
      ...choices,
    });
    expect(await job.undoApply(choices)).toBe(true);
    expect(deps.undoDatasetImport).toHaveBeenLastCalledWith(ID, {
      dry_run: false,
      ...choices,
    });
    expect(job.status).toBe('undoing');
    expect(job.undoReport).toBeNull();
    job.stop();
  });

  it('a served next step runs as served', async () => {
    const { job, deps } = make([jobFixture()]);
    const step = {
      action: 'cluster',
      method: 'POST',
      path: '/regions/cluster',
      reason: 'r',
    };
    expect(await job.runNextStep(step)).toBe(true);
    expect(deps.runServedNextStep).toHaveBeenCalledWith(step);
  });

  it('issues and entries pass their filters and pages', async () => {
    const { job, deps } = make([jobFixture()]);
    job.issueCode = 'label_file_missing';
    await job.loadIssues(2);
    expect(deps.getDatasetImportIssues).toHaveBeenCalledWith(ID, {
      code: 'label_file_missing',
      page: 2,
      page_size: 20,
    });
    job.entryFilters = { split: 'test', label_state: '', status: 'failed' };
    await job.loadEntries(1);
    expect(deps.getDatasetImportEntries).toHaveBeenCalledWith(ID, {
      split: 'test',
      label_state: null,
      status: 'failed',
      page: 1,
      page_size: 20,
    });
  });
});
