import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import type { SettingsAxisSpec } from '$lib/curationSettings';

const mocks = vi.hoisted(() => ({
  saveDefault: vi.fn(),
  clearDefault: vi.fn(),
  refresh: vi.fn(),
  reset: vi.fn(),
  success: vi.fn(),
  error: vi.fn(),
}));

vi.mock('$stores/curationSettings.svelte', () => ({
  curationSettingsStore: {
    settings: null,
    error: 'load error',
    saveDefault: mocks.saveDefault,
    clearDefault: mocks.clearDefault,
    refresh: mocks.refresh,
  },
}));
vi.mock('$stores/strategies.svelte', () => ({
  strategiesStore: { methods: null, reset: mocks.reset, init: vi.fn() },
}));
vi.mock('$stores/toast.svelte', () => ({
  toastStore: { success: mocks.success, error: mocks.error },
}));

import { createDefaultsEditor } from './curationSettingsEditor.svelte';
import { curationSettingsStore } from '$stores/curationSettings.svelte';

const spec = { axis: 'vlm', label: 'VLM' } as unknown as SettingsAxisSpec;

beforeEach(() => {
  vi.clearAllMocks();
  (curationSettingsStore as { error: string | null }).error = 'load error';
});

function pending() {
  const ed = createDefaultsEditor();
  ed.onSelectChange(spec, 'model-a');
  ed.openConfirm(spec);
  return ed;
}

describe('createDefaultsEditor.confirmAction', () => {
  it('does nothing without an open confirm', async () => {
    const ed = createDefaultsEditor();
    await ed.confirmAction();
    expect(mocks.saveDefault).not.toHaveBeenCalled();
    expect(mocks.clearDefault).not.toHaveBeenCalled();
  });

  it('saves the pending pick, resets caches, toasts and closes the dialog', async () => {
    mocks.saveDefault.mockResolvedValue(undefined);
    const ed = pending();
    await ed.confirmAction();
    expect(mocks.saveDefault).toHaveBeenCalledWith('vlm', 'model-a');
    expect(mocks.reset).toHaveBeenCalledTimes(1);
    expect(mocks.success).toHaveBeenCalledWith('Shared VLM default set to model-a');
    expect(ed.confirmSpec).toBeNull();
    expect(ed.pendingId).toBeNull();
    expect(ed.confirmMode).toBe('set');
    expect(ed.saveErrors).toEqual({});
  });

  it('clears the default in clear mode', async () => {
    mocks.clearDefault.mockResolvedValue(undefined);
    const ed = createDefaultsEditor();
    ed.openConfirmClear(spec);
    await ed.confirmAction();
    expect(mocks.clearDefault).toHaveBeenCalledWith('vlm');
    expect(mocks.success).toHaveBeenCalledWith('Shared VLM default cleared');
  });

  it('shows the served unknown-strategy words, re-syncs on 422 and drops the pick', async () => {
    mocks.saveDefault.mockRejectedValue(
      new ApiError(422, '/x', {
        detail: { axis: 'detection_profile', requested: 'zz', valid_ids: ['a', 'b'] },
      }),
    );
    const ed = pending();
    await ed.confirmAction();
    const msg = 'Unknown detection profile "zz" — valid: a, b.';
    expect(ed.saveErrors).toEqual({ vlm: msg });
    expect(mocks.error).toHaveBeenCalledWith(msg);
    expect(curationSettingsStore.error).toBeNull();
    expect(mocks.reset).toHaveBeenCalledTimes(1);
    expect(mocks.refresh).toHaveBeenCalledTimes(1);
    expect(ed.confirmSpec).toBeNull();
  });

  it('flags an unacknowledged external refusal and shows its served message', async () => {
    mocks.saveDefault.mockRejectedValue(
      new ApiError(409, '/x', {
        detail: { error: 'vlm_external_not_acknowledged', message: 'Acknowledge first' },
      }),
    );
    const ed = pending();
    await ed.confirmAction();
    expect(ed.saveErrors).toEqual({ vlm: 'Acknowledge first' });
    expect(ed.ackRefused).toEqual({ vlm: true });
    expect(mocks.refresh).not.toHaveBeenCalled();
  });

  it('falls back to the generic error text without touching caches on a non-422', async () => {
    mocks.saveDefault.mockRejectedValue(new ApiError(500, '/x', { detail: 'boom' }));
    const ed = pending();
    await ed.confirmAction();
    expect(ed.saveErrors).toEqual({ vlm: 'boom' });
    expect(mocks.reset).not.toHaveBeenCalled();
  });
});
