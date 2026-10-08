/**
 * The region-profile list state: the served list (with templates) and
 * active profile, Rollback / Turn off with `expected_active`, the served
 * impact on demand (dropped once the activation changes), the vocabulary
 * for the read-only panel, Delete / Clone, the `detection_profile`
 * wake-up, and the `/health` re-poll after a successful write.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import type { CurationEvent } from '$lib/sse';
import {
  impactFixture,
  profileActiveFixture,
  profileDocFixture,
  profileListFixture,
  vocabularyFixture,
} from '$lib/test/fixtures/regionProfiles';
import { healthStore } from '$stores/health.svelte';
import {
  createProfileList,
  isProfileConfigEvent,
  type ProfileListDeps,
} from './profileListController.svelte';

function refusal(status: number, detail: Record<string, unknown>): ApiError {
  return new ApiError(status, '/x', { detail });
}

afterEach(() => vi.restoreAllMocks());

function setup(over: Partial<ProfileListDeps> = {}) {
  let emit: (e: CurationEvent) => void = () => {};
  const deps = {
    listRegionProfiles: vi.fn().mockResolvedValue(profileListFixture()),
    getActiveRegionProfile: vi.fn().mockResolvedValue(profileActiveFixture()),
    rollbackRegionProfile: vi.fn().mockResolvedValue(profileActiveFixture()),
    deactivateRegionProfile: vi
      .fn()
      .mockResolvedValue(
        profileActiveFixture({ active: { name: null, revision: null } }),
      ),
    activateRegionProfile: vi.fn(),
    deleteRegionProfile: vi.fn().mockResolvedValue(undefined),
    cloneRegionProfile: vi.fn().mockResolvedValue(profileDocFixture({ name: 'copy' })),
    getRegionProfileImpact: vi.fn().mockResolvedValue(impactFixture()),
    getConfigVocabulary: vi.fn().mockResolvedValue(vocabularyFixture()),
    onchanged: vi.fn(),
    subscribe: vi.fn((cb: (e: CurationEvent) => void) => {
      emit = cb;
      return { close: vi.fn() };
    }),
    ...over,
  };
  const list = createProfileList(deps);
  return { list, deps, emit: (e: CurationEvent) => emit(e) };
}

describe('ProfileList', () => {
  it('loads the served list and active profile', async () => {
    const { list } = setup();
    await list.load();
    expect(list.list?.profiles.map((p) => p.name)).toEqual(['env_tags', 'widget_tag']);
    expect(list.list?.templates?.[0]?.path).toBe(
      'examples/region_profiles/widget_tag.json',
    );
    expect(list.active.active?.axis).toBe('detection_profile');
  });

  it('Turn off sends expected_active, re-reads, polls health and drops the impact', async () => {
    const { list, deps } = setup();
    await list.load();
    await list.loadImpact();
    expect(list.impact).not.toBeNull();
    expect(await list.deactivate()).toBe(true);
    expect(deps.deactivateRegionProfile).toHaveBeenCalledWith({
      expected_active: { name: 'widget_tag', revision: 2 },
    });
    expect(deps.onchanged).toHaveBeenCalledTimes(1);
    expect(deps.listRegionProfiles).toHaveBeenCalledTimes(2);
    expect(list.impact).toBeNull();
  });

  it('a refused Turn off keeps the impact and does not poll', async () => {
    const { list, deps } = setup({
      deactivateRegionProfile: vi
        .fn()
        .mockRejectedValue(refusal(409, { error: 'active_conflict', message: 'Moved.' })),
    });
    await list.load();
    await list.loadImpact();
    expect(await list.deactivate()).toBe(false);
    expect(list.active.actionError).toBe('Moved.');
    expect(list.impact).not.toBeNull();
    expect(deps.onchanged).not.toHaveBeenCalled();
  });

  it('rollback drops a shown impact', async () => {
    const { list, deps } = setup();
    await list.load();
    await list.loadImpact();
    expect(await list.rollback()).toBe(true);
    expect(deps.rollbackRegionProfile).toHaveBeenCalledWith({
      expected_active: { name: 'widget_tag', revision: 2 },
    });
    expect(list.impact).toBeNull();
  });

  it('impact: served counts, or the served message on failure', async () => {
    const { list } = setup({
      getRegionProfileImpact: vi
        .fn()
        .mockRejectedValue(
          refusal(503, { error: 'config_store_unavailable', message: 'Down.' }),
        ),
    });
    await list.loadImpact();
    expect(list.impact).toBeNull();
    expect(list.impactError).toBe('Down.');
  });

  it('the vocabulary is read once, without other projects', async () => {
    const { list, deps } = setup();
    await list.loadVocabulary();
    await list.loadVocabulary();
    expect(deps.getConfigVocabulary).toHaveBeenCalledTimes(1);
    expect(deps.getConfigVocabulary).toHaveBeenCalledWith(false);
    expect(list.vocabulary?.segmenters[0]?.max_candidates).toBe(128);
  });

  it('delete sends the row revision; in_use shows the served message', async () => {
    const { list, deps } = setup({
      deleteRegionProfile: vi
        .fn()
        .mockRejectedValue(refusal(409, { error: 'in_use', message: 'It is active.' })),
    });
    const row = profileListFixture().profiles[1]!;
    expect(await list.remove(row)).toBe(false);
    expect(deps.deleteRegionProfile).toHaveBeenCalledWith('widget_tag', 3);
    expect(list.deleteError).toBe('It is active.');
  });

  it('clone from a template sends source "template"', async () => {
    const { list, deps } = setup();
    const doc = await list.clone(
      { name: 'widget_tag', source: 'template' },
      ' tags2 ',
      '',
    );
    expect(doc?.name).toBe('copy');
    expect(deps.cloneRegionProfile).toHaveBeenCalledWith('widget_tag', {
      new_name: 'tags2',
      revision: null,
      source: 'template',
      description: null,
    });
  });

  it('re-reads only on config.changed for the detection_profile axis', async () => {
    const { list, deps, emit } = setup();
    list.start();
    await vi.waitFor(() => expect(deps.listRegionProfiles).toHaveBeenCalledTimes(1));
    emit({ type: 'config.changed', axis: 'prompt_pack' } as unknown as CurationEvent);
    emit({
      type: 'config.changed',
      axis: 'detection_profile',
    } as unknown as CurationEvent);
    await vi.waitFor(() => expect(deps.listRegionProfiles).toHaveBeenCalledTimes(2));
    list.stop();
    expect(
      isProfileConfigEvent({ type: 'config.changed', axis: 'region_profile' } as never),
    ).toBe(false);
  });

  it('by default a successful write re-polls /health (the reload notice path)', async () => {
    const poll = vi.spyOn(healthStore, 'poll').mockResolvedValue();
    const { onchanged: _unused, ...deps } = setup().deps;
    void _unused;
    const list = createProfileList(deps);
    await list.load();
    await list.deactivate();
    expect(poll).toHaveBeenCalledTimes(1);
  });
});
