/** `runExport`: the synchronous export response is rendered as served. */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { exportStatus, exportYolo, previewExportYolo } from '$lib/api_export';
import { ApiError } from '$lib/api_errors';
import { toastStore } from '$stores/toast.svelte';
import { createExportRunController } from './exportRunController.svelte';

vi.mock('$lib/api_export', () => ({
  exportYolo: vi.fn(),
  exportStatus: vi.fn(),
  previewExportYolo: vi.fn(),
}));

beforeEach(() => {
  vi.resetAllMocks();
  vi.spyOn(toastStore, 'error').mockImplementation(() => undefined as never);
  vi.spyOn(toastStore, 'success').mockImplementation(() => undefined as never);
  vi.spyOn(toastStore, 'info').mockImplementation(() => undefined as never);
});

describe('refreshPreview', () => {
  const small = {
    image_count: 1,
    object_count: 1,
    images_dropped_not_fully_labeled: 38,
    images_with_unlabeled_items: 0,
    unlabeled_items_on_exported_images: 0,
    warning: 'Only 1 image would be exported; 38 are left out.',
  };

  it('asks the server for the counts of the current options and serves its warning', async () => {
    vi.mocked(previewExportYolo).mockResolvedValue(small);
    const c = createExportRunController(async () => {});
    c.requireFullyLabeled = true;
    await c.refreshPreview();
    expect(previewExportYolo).toHaveBeenCalledWith(
      expect.objectContaining({
        require_fully_labeled_images: true,
        collapse_near_duplicates: false,
      }),
    );
    expect(c.preview?.warning).toContain('Only 1 image');
    c.dispose();
  });

  it('drops a stale answer when a newer one was asked for', async () => {
    let release: (v: typeof small) => void = () => {};
    vi.mocked(previewExportYolo)
      .mockImplementationOnce(() => new Promise((r) => (release = r)))
      .mockResolvedValueOnce({ ...small, image_count: 9, warning: null });
    const c = createExportRunController(async () => {});
    const first = c.refreshPreview();
    await c.refreshPreview();
    release(small);
    await first;
    expect(c.preview?.image_count).toBe(9);
    c.dispose();
  });

  it('a failed preview shows the failure instead of a count, and never blocks the export', async () => {
    vi.mocked(previewExportYolo).mockRejectedValue(new Error('down'));
    const c = createExportRunController(async () => {});
    await c.refreshPreview();
    expect(c.preview).toBeNull();
    expect(c.previewError).toBe('down');
    c.dispose();
  });

  it('a 409 near_dup_stale shows the stale levels, not the count it had before', async () => {
    vi.mocked(previewExportYolo).mockResolvedValueOnce(small);
    const c = createExportRunController(async () => {});
    await c.refreshPreview();
    expect(c.preview).not.toBeNull();
    const levels = [{ level: 'item', stale_reasons: ['no_run'], pending_count: 152 }];
    vi.mocked(previewExportYolo).mockRejectedValueOnce(
      new ApiError(409, '/export/yolo/preview', {
        detail: { code: 'near_dup_stale', levels },
      }),
    );
    await c.refreshPreview();
    expect(c.preview).toBeNull();
    expect(c.previewError).toBeNull();
    expect(c.nearDup.stale).toEqual(levels);
    vi.mocked(previewExportYolo).mockResolvedValueOnce(small);
    await c.refreshPreview();
    expect(c.nearDup.stale).toEqual([]);
    expect(c.preview).not.toBeNull();
    c.dispose();
  });
});

describe('runExport', () => {
  it('a finished export maps the response onto exportState and reloads the page', async () => {
    vi.mocked(exportYolo).mockResolvedValue({
      status: 'success',
      finished_at: '2026-01-02T00:00:00Z',
      export_dir: '/out/v1',
      image_count: 7,
      object_count: 9,
    } as never);
    const loadAll = vi.fn(async () => {});
    const c = createExportRunController(loadAll);
    c.versionTag = ' v1 ';
    await c.runExport();
    expect(exportYolo).toHaveBeenCalledWith(
      expect.objectContaining({ version_tag: 'v1', require_fully_labeled_images: false }),
    );
    expect(c.exportState).toMatchObject({
      status: 'success',
      last_run: '2026-01-02T00:00:00Z',
      export_dir: '/out/v1',
      error: null,
      image_count: 7,
      object_count: 9,
      split_object_counts: null,
      skipped_items: null,
    });
    expect(c.exportRunning).toBe(false);
    expect(c.exportModalOpen).toBe(true);
    expect(toastStore.success).toHaveBeenCalledWith('Export complete: /out/v1');
    expect(loadAll).toHaveBeenCalledTimes(1);
    c.dispose();
  });

  it('a failed export toasts the served message and records it as the error', async () => {
    vi.mocked(exportYolo).mockResolvedValue({
      status: 'failed',
      message: 'disk full',
    } as never);
    const c = createExportRunController(async () => {});
    await c.runExport();
    expect(c.exportState).toMatchObject({ status: 'failed', error: 'disk full' });
    expect(toastStore.error).toHaveBeenCalledWith('Export failed: disk full');
    expect(c.exportRunning).toBe(false);
  });

  it('an unrecognised status is shown as an info toast', async () => {
    vi.mocked(exportYolo).mockResolvedValue({ status: 'idle' } as never);
    const c = createExportRunController(async () => {});
    await c.runExport();
    expect(toastStore.info).toHaveBeenCalledWith('Export: idle');
    expect(c.exportRunning).toBe(false);
  });

  it('a still-running export keeps running and starts the status poll', async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(exportYolo).mockResolvedValue({ status: 'running' } as never);
      vi.mocked(exportStatus).mockResolvedValue({
        status: 'failed',
        error: 'x',
      } as never);
      const c = createExportRunController(async () => {});
      await c.runExport();
      expect(toastStore.info).toHaveBeenCalledWith('Export started: running');
      expect(c.exportRunning).toBe(true);
      await vi.advanceTimersByTimeAsync(5000);
      expect(exportStatus).toHaveBeenCalledTimes(1);
      expect(c.exportRunning).toBe(false);
      expect(toastStore.error).toHaveBeenCalledWith('Export failed: x');
      c.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it('a refused request toasts the served detail and stops running', async () => {
    vi.mocked(exportYolo).mockRejectedValue(
      new ApiError(422, '/x', { detail: 'nothing to export: empty' }),
    );
    const c = createExportRunController(async () => {});
    await c.runExport();
    expect(toastStore.error).toHaveBeenCalledWith(
      'Export failed: nothing to export: empty',
    );
    expect(c.exportRunning).toBe(false);
  });
});
