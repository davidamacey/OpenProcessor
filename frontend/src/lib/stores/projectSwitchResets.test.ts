/**
 * What a project switch does to the per-project stores (review §3.8,
 * §7.4): the region profile re-seeds for the new project with NO
 * "reload to apply" notice (a different project's profile is not a
 * change), a vocabulary load started for the old project never lands in
 * the new one, and the class registry is dropped.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from '$lib/api';
import { projectsStore } from '$stores/projects.svelte';
import {
  REGION_PROFILE_CHANGED_NOTICE,
  loadRegionProfile,
  regionProfileStore,
} from '$stores/regionProfile.svelte';
import { regionStatusesStore } from '$stores/regionStatuses.svelte';
import { classesStore } from '$stores/classes.svelte';
import { toastStore } from '$stores/toast.svelte';
import { testProject } from '$lib/test/fixtures/projects';
import type { ApiHealth } from '$lib/types';

const DEFAULT = testProject({ slug: 'default', prefix: API_PREFIX, is_default: true });
const BETA = testProject({ slug: 'beta' });

const profile = (name: string) => ({
  name,
  display_name: `${name} regions`,
  display_name_singular: name,
  region_class_name: name,
  text_reader: '',
  reads_text: false,
  text_hint_enabled: false,
});

afterEach(() => {
  vi.unstubAllGlobals();
  projectsStore.select(DEFAULT);
  regionProfileStore.reset();
  toastStore.toasts = [];
});

describe('project switch resets', () => {
  it('re-seeds the region profile for the new project without a reload notice', async () => {
    await loadRegionProfile(
      async () => ({ status: 'ok', region_profile: profile('tag') }) as ApiHealth,
    );
    expect(regionProfileStore.profile?.name).toBe('tag');

    projectsStore.select(BETA);
    expect(regionProfileStore.loaded).toBe(false);
    expect(regionProfileStore.profile).toBeNull();

    await loadRegionProfile(
      async () => ({ status: 'ok', region_profile: profile('label') }) as ApiHealth,
    );
    expect(regionProfileStore.profile?.name).toBe('label');
    expect(regionProfileStore.changed).toBe(false);
    expect(toastStore.toasts.map((t) => t.text)).not.toContain(
      REGION_PROFILE_CHANGED_NOTICE,
    );
  });

  it('still raises the notice for a different profile on the SAME project', async () => {
    await loadRegionProfile(
      async () => ({ status: 'ok', region_profile: profile('tag') }) as ApiHealth,
    );
    regionProfileStore.observe(profile('other'));
    expect(regionProfileStore.changed).toBe(true);
  });

  it('drops a vocabulary load that was started for the previous project', async () => {
    let release!: (r: Response) => void;
    vi.stubGlobal(
      'fetch',
      vi.fn().mockReturnValueOnce(new Promise<Response>((r) => (release = r))),
    );
    const pending = regionStatusesStore.init();

    projectsStore.select(BETA);
    release(
      new Response(
        JSON.stringify({ statuses: [{ value: 'old_project_status', label: 'Old' }] }),
        {
          headers: { 'content-type': 'application/json' },
        },
      ),
    );
    await pending;
    expect(regionStatusesStore.list).toEqual([]);
    expect(regionStatusesStore.loaded).toBe(false);
  });

  it('drops the class registry', () => {
    classesStore.classes = [{ id: 1, name: 'old_project_class' } as never];
    projectsStore.select(BETA);
    expect(classesStore.classes).toEqual([]);
  });
});
