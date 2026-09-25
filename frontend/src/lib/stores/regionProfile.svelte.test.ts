/**
 * The served region profile gate (docs/design/domain-neutral-audit-2026-09-24.md
 * §5.2): seeded once from `{API_PREFIX}/health`, fail-closed, and a later
 * change or a region route's 409 becomes one "reload" notice rather than
 * an error toast.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  API_PREFIX,
  apiFetch,
  RegionProfileUnavailableError,
  type ApiError,
} from '$lib/api';
import {
  registeredSlots,
  resetDeploymentSlots,
  slotForClassName,
} from '$lib/annotations/registeredSlots';
import { REGION_PROFILE_UNAVAILABLE_MESSAGE } from '$lib/regionProfileUnavailable';
import { WIDGET_TAG_CLASS, WIDGET_TAG_PROFILE } from '$lib/test/fixtures/regionSlot';
import type { ApiHealth } from '$lib/types';
import { toastStore } from './toast.svelte';
import {
  loadRegionProfile,
  REGION_PROFILE_CHANGED_NOTICE,
  regionProfileStore,
} from './regionProfile.svelte';

function health(over: Partial<ApiHealth> = {}): ApiHealth {
  return { status: 'ok', ...over };
}

beforeEach(() => {
  regionProfileStore.reset();
  resetDeploymentSlots();
  toastStore.toasts = [];
});

afterEach(() => {
  regionProfileStore.reset();
  resetDeploymentSlots();
  toastStore.toasts = [];
  vi.unstubAllGlobals();
});

describe('loadRegionProfile', () => {
  it('seeds the store and registers the region slot from a served profile', async () => {
    const fetchHealth = vi
      .fn()
      .mockResolvedValue(health({ region_profile: WIDGET_TAG_PROFILE }));
    await loadRegionProfile(fetchHealth);
    expect(regionProfileStore.configured).toBe(true);
    expect(regionProfileStore.profile).toEqual(WIDGET_TAG_PROFILE);
    expect(slotForClassName(WIDGET_TAG_CLASS)?.key).toBe(WIDGET_TAG_PROFILE.name);
  });

  it('region_profile: null means not configured and no slot', async () => {
    await loadRegionProfile(vi.fn().mockResolvedValue(health({ region_profile: null })));
    expect(regionProfileStore.loaded).toBe(true);
    expect(regionProfileStore.configured).toBe(false);
    expect(registeredSlots).toEqual([]);
  });

  it('a backend that predates the field counts as not configured', async () => {
    await loadRegionProfile(vi.fn().mockResolvedValue(health()));
    expect(regionProfileStore.configured).toBe(false);
    expect(registeredSlots).toEqual([]);
  });

  it('fails closed on a failed /health', async () => {
    await loadRegionProfile(vi.fn().mockRejectedValue(new Error('down')));
    expect(regionProfileStore.loaded).toBe(true);
    expect(regionProfileStore.configured).toBe(false);
    expect(registeredSlots).toEqual([]);
  });

  it('reads the prefixed curation health route with a bounded signal', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(health({ region_profile: WIDGET_TAG_PROFILE })), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    await loadRegionProfile();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/health`);
    expect((init as RequestInit).signal).toBeInstanceOf(AbortSignal);
    expect(regionProfileStore.configured).toBe(true);
  });

  it('is memoized: a second call does not refetch', async () => {
    const fetchHealth = vi
      .fn()
      .mockResolvedValue(health({ region_profile: WIDGET_TAG_PROFILE }));
    await loadRegionProfile(fetchHealth);
    await loadRegionProfile(fetchHealth);
    expect(fetchHealth).toHaveBeenCalledTimes(1);
  });
});

describe('observe (later /health polls)', () => {
  it('the same profile shows nothing', () => {
    regionProfileStore.seed(WIDGET_TAG_PROFILE);
    regionProfileStore.observe({ ...WIDGET_TAG_PROFILE });
    expect(toastStore.toasts).toEqual([]);
  });

  it('a changed profile shows one sticky reload notice, however many polls disagree', () => {
    regionProfileStore.seed(WIDGET_TAG_PROFILE);
    regionProfileStore.observe(null);
    regionProfileStore.observe({ ...WIDGET_TAG_PROFILE, display_name: 'Other' });
    expect(toastStore.toasts.map((t) => t.text)).toEqual([REGION_PROFILE_CHANGED_NOTICE]);
    expect(toastStore.toasts[0].ttl_ms).toBe(0);
    // The UI keeps the profile it was built from.
    expect(regionProfileStore.profile).toEqual(WIDGET_TAG_PROFILE);
  });

  it.each([
    ['name', { name: 'other_tag' }],
    ['display_name', { display_name: 'Other' }],
    ['region_class_name', { region_class_name: 'other_class' }],
    ['text_reader', { text_reader: 'vlm' }],
  ])('a change to %s alone is a change', (_field, over) => {
    regionProfileStore.seed(WIDGET_TAG_PROFILE);
    regionProfileStore.observe({ ...WIDGET_TAG_PROFILE, ...over });
    expect(toastStore.toasts.map((t) => t.text)).toEqual([REGION_PROFILE_CHANGED_NOTICE]);
  });

  it('a profile appearing where there was none is a change too', () => {
    regionProfileStore.seed(null);
    regionProfileStore.observe(WIDGET_TAG_PROFILE);
    expect(toastStore.toasts).toHaveLength(1);
  });

  it('does nothing before the store is seeded', () => {
    regionProfileStore.observe(WIDGET_TAG_PROFILE);
    expect(toastStore.toasts).toEqual([]);
  });
});

describe("a region route's 409 'no region profile is configured'", () => {
  function conflict(): Response {
    return new Response(JSON.stringify({ detail: 'no region profile is configured' }), {
      status: 409,
      headers: { 'content-type': 'application/json' },
    });
  }

  it('throws RegionProfileUnavailableError (no retry) and triggers the reload notice', async () => {
    regionProfileStore.seed(WIDGET_TAG_PROFILE);
    const fetchMock = vi.fn().mockResolvedValue(conflict());
    vi.stubGlobal('fetch', fetchMock);
    const err = await apiFetch(`${API_PREFIX}/regions`).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(RegionProfileUnavailableError);
    expect((err as ApiError).status).toBe(409);
    expect((err as Error).message).toBe(REGION_PROFILE_UNAVAILABLE_MESSAGE);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(toastStore.toasts.map((t) => t.text)).toEqual([REGION_PROFILE_CHANGED_NOTICE]);
  });

  it("a call site's own error toast for it is swallowed", () => {
    toastStore.error(`Save failed: ${REGION_PROFILE_UNAVAILABLE_MESSAGE}`);
    expect(toastStore.toasts).toEqual([]);
    toastStore.error('Save failed: something else');
    expect(toastStore.toasts).toHaveLength(1);
  });

  it('any other 409 is an ordinary ApiError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'class name already exists' }), {
          status: 409,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    );
    const err = await apiFetch(`${API_PREFIX}/classes`).catch((e: unknown) => e);
    expect(err).not.toBeInstanceOf(RegionProfileUnavailableError);
    expect((err as ApiError).status).toBe(409);
    expect((err as Error).message).toContain('class name already exists');
  });
});
