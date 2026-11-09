/**
 * A 409 `near_dup_stale` from the holdout freeze or the export is shown as the
 * stale-levels panel, not as a failure toast; any other error still toasts.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import { exportYolo, freezeTestHoldout } from '$lib/api_export';
import { toastStore } from '$stores/toast.svelte';
import { createExportRunController } from './exportRunController.svelte';
import { createFreezeHoldoutController } from './freezeHoldoutController.svelte';

vi.mock('$lib/api_export', () => ({
  exportYolo: vi.fn(),
  exportStatus: vi.fn(),
  freezeTestHoldout: vi.fn(),
}));

const stale = new ApiError(409, '/x', {
  detail: {
    code: 'near_dup_stale',
    levels: [{ level: 'item', stale_reasons: ['rebuild_required'], pending_count: 0 }],
  },
});

beforeEach(() => {
  vi.resetAllMocks();
  vi.spyOn(window, 'confirm').mockReturnValue(true);
  vi.spyOn(toastStore, 'error').mockImplementation(() => undefined as never);
  vi.spyOn(toastStore, 'success').mockImplementation(() => undefined as never);
});

describe('freeze', () => {
  it('shows the stale levels without an error toast and keeps the modal open', async () => {
    vi.mocked(freezeTestHoldout).mockRejectedValue(stale);
    const freeze = createFreezeHoldoutController(async () => {});
    freeze.openFreeze();
    await freeze.submitFreeze();
    expect(freeze.nearDup.stale.map((l) => l.level)).toEqual(['item']);
    expect(freeze.freezeOpen).toBe(true);
    expect(toastStore.error).not.toHaveBeenCalled();
  });

  it('toasts any other failure and shows no panel', async () => {
    vi.mocked(freezeTestHoldout).mockRejectedValue(
      new ApiError(422, '/x', { detail: 'no' }),
    );
    const freeze = createFreezeHoldoutController(async () => {});
    freeze.openFreeze();
    await freeze.submitFreeze();
    expect(freeze.nearDup.stale).toEqual([]);
    expect(toastStore.error).toHaveBeenCalledTimes(1);
  });

  it('reports the items pulled in as near-duplicates', async () => {
    vi.mocked(freezeTestHoldout).mockResolvedValue({
      n_frozen: 10,
      n_classes_covered: 2,
      test_holdout_sha: 'x',
      per_class_counts: {},
      selection: 'sha1_per_class',
      percent: 10,
      min_per_class: 5,
      moved_by_near_dup: 3,
    });
    const freeze = createFreezeHoldoutController(async () => {});
    freeze.openFreeze();
    await freeze.submitFreeze();
    expect(vi.mocked(toastStore.success).mock.calls[0]![0]).toContain(
      '3 pulled in as near-duplicates',
    );
  });
});

describe('export', () => {
  it('shows the stale levels and closes the progress modal', async () => {
    vi.mocked(exportYolo).mockRejectedValue(stale);
    const run = createExportRunController(async () => {});
    await run.runExport();
    expect(run.nearDup.stale.map((l) => l.level)).toEqual(['item']);
    expect(run.exportModalOpen).toBe(false);
    expect(run.exportRunning).toBe(false);
    expect(toastStore.error).not.toHaveBeenCalled();
    run.dispose();
  });

  it('sends collapse_near_duplicates only when the box is ticked', async () => {
    vi.mocked(exportYolo).mockResolvedValue({ status: 'success' } as never);
    const run = createExportRunController(async () => {});
    await run.runExport();
    expect(vi.mocked(exportYolo).mock.calls[0]![0]).toMatchObject({
      collapse_near_duplicates: false,
    });
    run.collapseNearDuplicates = true;
    await run.runExport();
    expect(vi.mocked(exportYolo).mock.calls[1]![0]).toMatchObject({
      collapse_near_duplicates: true,
    });
    run.dispose();
  });
});
