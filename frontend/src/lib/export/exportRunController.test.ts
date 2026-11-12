/** `runExport`: the synchronous export response is rendered as served. */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { exportStatus, exportYolo } from '$lib/api_export';
import { ApiError } from '$lib/api_errors';
import { toastStore } from '$stores/toast.svelte';
import { createExportRunController } from './exportRunController.svelte';

vi.mock('$lib/api_export', () => ({
  exportYolo: vi.fn(),
  exportStatus: vi.fn(),
}));

beforeEach(() => {
  vi.resetAllMocks();
  vi.spyOn(toastStore, 'error').mockImplementation(() => undefined as never);
  vi.spyOn(toastStore, 'success').mockImplementation(() => undefined as never);
  vi.spyOn(toastStore, 'info').mockImplementation(() => undefined as never);
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
