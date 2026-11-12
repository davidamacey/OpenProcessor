import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import type { PresetDoc } from '$lib/types_presets';
import { createPresetList } from './presetListController.svelte';
import { presetKindBySlug } from './presetKinds';

const kind = presetKindBySlug('near-dup-profiles')!;

const doc = (name: string, revision = 1, over: Partial<PresetDoc> = {}): PresetDoc => ({
  kind: 'near_dup_profile',
  name,
  revision,
  description: `${name} d`,
  body: { version: 1, item: { dedup_cosine: 0.97 } },
  created_at: '2026-11-01T00:00:00Z',
  updated_at: '2026-11-02T00:00:00Z',
  updated_by: 'api',
  ...over,
});

function setup(over = {}) {
  const deps = {
    listPresets: vi.fn().mockResolvedValue({ items: [doc('strict', 2)] }),
    getPreset: vi.fn().mockResolvedValue(doc('strict', 2)),
    putPreset: vi.fn().mockImplementation(async (_k, name: string) => doc(name)),
    deletePreset: vi.fn().mockResolvedValue(undefined),
    getActivePreset: vi.fn().mockResolvedValue({
      axis: 'near_dup',
      state: 'active',
      name: 'strict',
      revision: 2,
      activated_at: null,
      setting_revision: null,
    }),
    ...over,
  };
  return { list: createPresetList(kind, deps), deps };
}

describe('PresetList', () => {
  it('loads the served documents and the active refs', async () => {
    const { list } = setup();
    await list.load();
    expect(list.list?.items.map((d) => d.name)).toEqual(['strict']);
    expect(list.active.activeOn('strict')).toEqual(['near_dup']);
  });

  it('clone copies the served document under the new name as a create', async () => {
    const { list, deps } = setup();
    await list.load();
    const made = await list.clone({ name: 'strict', source: null }, ' loose ', '');
    expect(made?.name).toBe('loose');
    expect(deps.getPreset).toHaveBeenCalledWith('near_dup_profile', 'strict');
    expect(deps.putPreset).toHaveBeenCalledWith('near_dup_profile', 'loose', {
      body: { version: 1, item: { dedup_cosine: 0.97 } },
      description: 'strict d',
      expected_revision: null,
    });
  });

  it('create writes the operator body, reloads, and shows a refusal with its reasons', async () => {
    const { list, deps } = setup();
    await list.load();
    const made = await list.create(' new ', ' words ', { version: 1 });
    expect(made?.name).toBe('new');
    expect(deps.putPreset).toHaveBeenCalledWith('near_dup_profile', 'new', {
      body: { version: 1 },
      description: 'words',
      expected_revision: null,
    });
    expect(deps.listPresets).toHaveBeenCalledTimes(2);

    deps.putPreset.mockRejectedValueOnce(
      new ApiError(422, '/x', {
        detail: {
          error: 'validation_failed',
          message: 'invalid config body: 1 error(s)',
          reasons: ['item.leak_cosine: leak_cosine may not exceed dedup_cosine'],
        },
      }),
    );
    expect(await list.create('bad', '', {})).toBeNull();
    expect(list.createError).toBe(
      'invalid config body: 1 error(s): item.leak_cosine: leak_cosine may not exceed dedup_cosine',
    );
  });

  it('delete sends the revision the list served; a stale one reloads', async () => {
    const { list, deps } = setup();
    await list.load();
    expect(await list.remove({ name: 'strict', revision: 2 })).toBe(true);
    expect(deps.deletePreset).toHaveBeenCalledWith('near_dup_profile', 'strict', 2);

    deps.deletePreset.mockRejectedValueOnce(
      new ApiError(409, '/x', {
        detail: { error: 'revision_conflict', message: "'strict' is at revision 3" },
      }),
    );
    const loads = deps.listPresets.mock.calls.length;
    expect(await list.remove({ name: 'strict', revision: 2 })).toBe(false);
    expect(list.deleteError).toBe("'strict' is at revision 3");
    expect(deps.listPresets.mock.calls.length).toBe(loads + 1);
  });
});
