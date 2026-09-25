import { afterEach, describe, expect, it, vi } from 'vitest';
import { healthChip, healthStore, HEALTH_CHIP_TEXT } from './health.svelte';
import {
  REGION_PROFILE_CHANGED_NOTICE,
  regionProfileStore,
} from './regionProfile.svelte';
import { toastStore } from './toast.svelte';
import { WIDGET_TAG_PROFILE } from '$lib/test/fixtures/regionSlot';

describe('healthChip', () => {
  it('reads "checking", not "down", before the first poll has settled', () => {
    expect(healthChip(false, null)).toBe('checking');
    expect(HEALTH_CHIP_TEXT[healthChip(false, null)]).not.toContain('down');
  });

  it('reports ok / down once a poll has settled', () => {
    expect(healthChip(true, 1)).toBe('ok');
    expect(healthChip(false, 1)).toBe('down');
  });
});

describe('healthStore.poll feeds the region-profile gate', () => {
  afterEach(() => {
    regionProfileStore.reset();
    toastStore.toasts = [];
    vi.unstubAllGlobals();
  });

  function serve(body: unknown): void {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    );
  }

  it('a poll that drops the seeded profile raises the reload notice', async () => {
    regionProfileStore.seed(WIDGET_TAG_PROFILE);
    serve({ status: 'ok', region_profile: null });
    await healthStore.poll();
    expect(toastStore.toasts.map((t) => t.text)).toEqual([REGION_PROFILE_CHANGED_NOTICE]);
  });

  it('a poll serving the same profile is silent', async () => {
    regionProfileStore.seed(WIDGET_TAG_PROFILE);
    serve({ status: 'ok', region_profile: WIDGET_TAG_PROFILE });
    await healthStore.poll();
    expect(toastStore.toasts).toEqual([]);
  });
});
