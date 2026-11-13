/**
 * The freeze modal is the only confirmation (no native `window.confirm`), and a refreeze of an
 * existing holdout sends `force`.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { freezeTestHoldout } from '$lib/api_export';
import { toastStore } from '$stores/toast.svelte';
import { createFreezeHoldoutController } from './freezeHoldoutController.svelte';

vi.mock('$lib/api_export', () => ({ freezeTestHoldout: vi.fn() }));

const FROZEN = {
  n_frozen: 4,
  n_classes_covered: 2,
  test_holdout_sha: 'x',
  per_class_counts: {},
  selection: 'sha1_per_class',
  min_per_class: 5,
  moved_by_near_dup: 0,
};

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(freezeTestHoldout).mockResolvedValue(FROZEN as never);
  vi.spyOn(toastStore, 'success').mockImplementation(() => undefined as never);
});

describe('submitFreeze', () => {
  it('asks nothing natively and sends no force on a first freeze', async () => {
    const confirm = vi.spyOn(window, 'confirm');
    const freeze = createFreezeHoldoutController(async () => {});
    freeze.openFreeze();
    await freeze.submitFreeze();
    expect(confirm).not.toHaveBeenCalled();
    expect(freezeTestHoldout).toHaveBeenCalledWith({ percent: 10, force: undefined });
  });

  it('a refreeze sends force', async () => {
    const freeze = createFreezeHoldoutController(async () => {});
    freeze.openFreeze({ refreeze: true });
    expect(freeze.freezeForce).toBe(true);
    await freeze.submitFreeze();
    expect(freezeTestHoldout).toHaveBeenCalledWith({ percent: 10, force: true });
  });

  it('reopening as a first freeze clears a previous refreeze', () => {
    const freeze = createFreezeHoldoutController(async () => {});
    freeze.openFreeze({ refreeze: true });
    freeze.closeFreeze();
    freeze.openFreeze();
    expect(freeze.freezeForce).toBe(false);
  });
});
