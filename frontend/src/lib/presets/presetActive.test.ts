import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import type { ActivePreset } from '$lib/types_presets';
import { PresetActive } from './presetActive.svelte';
import { PRESET_SCHEMAS } from '$lib/test/fixtures/presetSchemas';
import { PRESET_KINDS, presetKindBySlug } from './presetKinds';

const kind = (slug: string) => presetKindBySlug(slug)!;
const schemaOf = vi.fn(async (k: keyof typeof PRESET_SCHEMAS) => PRESET_SCHEMAS[k]);

const entry = (over: Partial<ActivePreset>): ActivePreset => ({
  axis: 'near_dup',
  state: 'none',
  name: null,
  revision: null,
  activated_at: null,
  setting_revision: null,
  ...over,
});

describe('PresetActive schema', () => {
  it('reads the served schema of the kind once, however often it reloads', async () => {
    const schema = vi.fn().mockResolvedValue(PRESET_SCHEMAS.near_dup_profile);
    const a = new PresetActive(kind('near-dup-profiles'), {
      getPresetSchema: schema,
      getActivePreset: vi.fn().mockResolvedValue(entry({})),
    });
    await a.load();
    await a.load();
    expect(schema).toHaveBeenCalledTimes(1);
    expect(schema).toHaveBeenCalledWith('near_dup_profile');
    expect(a.schema?.kind).toBe('near_dup_profile');
    expect(a.schemaError).toBeNull();
  });

  it('a schema that cannot be read is shown as served and does not block the active read', async () => {
    const a = new PresetActive(kind('near-dup-profiles'), {
      getPresetSchema: vi.fn().mockRejectedValue(
        new ApiError(500, '/x', {
          detail: { error: 'internal', message: 'schema broke' },
        }),
      ),
      getActivePreset: vi
        .fn()
        .mockResolvedValue(entry({ state: 'active', name: 'strict', revision: 2 })),
    });
    await a.load();
    expect(a.schema).toBeNull();
    expect(a.schemaError).toMatch(/schema broke/);
    expect(a.activeOn('strict')).toEqual(['near_dup']);
  });
});

describe('PresetActive', () => {
  it('reads every axis of a cluster profile, in the kind order', async () => {
    const read = vi.fn(async (_k: string, axis: string) =>
      entry({
        axis,
        state: 'active',
        name: axis === 'clustering_item' ? 'a' : 'b',
        revision: 1,
      }),
    );
    const a = new PresetActive(kind('cluster-profiles'), {
      getPresetSchema: schemaOf,
      getActivePreset: read as never,
    });
    await a.load();
    expect(read.mock.calls.map((c) => c[1])).toEqual([
      'clustering_item',
      'clustering_region_box',
    ]);
    expect(a.activeOn('a')).toEqual(['clustering_item']);
    expect(a.activeOn('b')).toEqual(['clustering_region_box']);
  });

  it('activates against the ref last read and reloads', async () => {
    const read = vi
      .fn()
      .mockResolvedValueOnce(entry({ state: 'active', name: 'old', revision: 1 }))
      .mockResolvedValueOnce(entry({ state: 'active', name: 'strict', revision: 3 }));
    const write = vi.fn().mockResolvedValue({});
    const a = new PresetActive(kind('near-dup-profiles'), {
      getPresetSchema: schemaOf,
      getActivePreset: read,
      activatePreset: write,
    });
    await a.load();
    expect(await a.activate('strict', 3, 'near_dup')).toBe(true);
    expect(write).toHaveBeenCalledWith('near_dup_profile', 'strict', {
      axis: 'near_dup',
      revision: 3,
      expected_active: { name: 'old', revision: 1 },
    });
    expect(a.activeOn('strict')).toEqual(['near_dup']);
  });

  it('sends a null expected_active when nothing is active', async () => {
    const write = vi.fn().mockResolvedValue({});
    const a = new PresetActive(kind('assist-loop-presets'), {
      getPresetSchema: schemaOf,
      getActivePreset: vi.fn().mockResolvedValue(entry({ axis: 'assist_loop_preset' })),
      activatePreset: write,
    });
    await a.load();
    await a.activate('p', 1, 'assist_loop_preset');
    expect(write.mock.calls[0]![2]).toMatchObject({ expected_active: null });
  });

  it('a stale activation shows the served 409 and re-reads', async () => {
    const read = vi
      .fn()
      .mockResolvedValue(entry({ state: 'active', name: 'x', revision: 1 }));
    const write = vi.fn().mockRejectedValue(
      new ApiError(409, '/x', {
        detail: { error: 'active_conflict', message: 'the active profile changed' },
      }),
    );
    const a = new PresetActive(kind('near-dup-profiles'), {
      getPresetSchema: schemaOf,
      getActivePreset: read,
      activatePreset: write,
    });
    await a.load();
    expect(await a.activate('y', 2, 'near_dup')).toBe(false);
    expect(a.actionError).toBe('the active profile changed');
    expect(a.errorDetail?.error).toBe('active_conflict');
    expect(read).toHaveBeenCalledTimes(2);
  });

  it('the serving profile is activated against the setting revision', async () => {
    const write = vi.fn().mockResolvedValue({});
    const a = new PresetActive(kind('serving-profiles'), {
      getPresetSchema: schemaOf,
      getActiveServingProfile: vi.fn().mockResolvedValue(
        entry({
          axis: null,
          state: 'active',
          name: 'one',
          revision: 2,
          setting_revision: 7,
        }),
      ),
      activateServingProfile: write,
    });
    await a.load();
    await a.activate('two', 1, null);
    expect(write).toHaveBeenCalledWith('two', { revision: 1, expected_revision: 7 });
  });

  it('a failed read keeps the served error and offers no activation', async () => {
    const a = new PresetActive(kind('near-dup-profiles'), {
      getPresetSchema: schemaOf,
      getActivePreset: vi.fn().mockRejectedValue(new Error('down')),
    });
    await a.load();
    expect(a.loadError).toBe('down');
    expect(await a.activate('x', 1, 'near_dup')).toBe(false);
  });

  it('every kind names its axes, and only the deployment kind has none', () => {
    expect(PRESET_KINDS.map((k) => [k.id, k.axes.length])).toEqual([
      ['serving_profile', 0],
      ['cluster_profile', 2],
      ['assist_loop_preset', 1],
      ['near_dup_profile', 1],
    ]);
  });
});
