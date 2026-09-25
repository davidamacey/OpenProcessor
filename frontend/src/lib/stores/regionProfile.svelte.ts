/**
 * RegionProfileStore — whether the backend has a region profile, and
 * which one (`GET {API_PREFIX}/health` `region_profile`, OpenProcessor
 * naming-w2; docs/design/domain-neutral-audit-2026-09-24.md §5.2).
 *
 * THE gate for every region feature: with no profile the registry holds
 * no region slot (so no region tab, gallery, inventory card, sub-box
 * editor or detections panel renders) and the root layout never calls a
 * region route.
 *
 * Seeded once, before first render, by `loadRegionProfile()` from the
 * root layout's `load()`. A timeout, a failure, or a backend too old to
 * serve the field all count as "not configured" — fail closed, so no
 * region route is ever called against a backend that would 409 it.
 *
 * The UI is built once from the seeded value. When a later `/health`
 * poll (or a region route's 409) says the profile changed, the store
 * shows one "reload to apply" notice rather than hot-swapping tabs.
 */

import { getHealth } from '$lib/api';
import { installServedRegionProfile } from '$lib/annotations/registeredSlots';
import { setRegionProfileUnavailableListener } from '$lib/regionProfileUnavailable';
import type { ServedRegionProfile } from '$lib/types';
import { toastStore } from '$stores/toast.svelte';

export const REGION_PROFILE_TIMEOUT_MS = 2000;

export const REGION_PROFILE_CHANGED_NOTICE =
  "The backend's region profile changed — reload the page to apply it.";

function normalize(
  p: ServedRegionProfile | null | undefined,
): ServedRegionProfile | null {
  if (!p || typeof p !== 'object' || typeof p.name !== 'string' || !p.name) return null;
  return {
    name: p.name,
    display_name: typeof p.display_name === 'string' ? p.display_name : '',
    display_name_singular:
      typeof p.display_name_singular === 'string' ? p.display_name_singular : '',
    region_class_name: typeof p.region_class_name === 'string' ? p.region_class_name : '',
    text_reader: typeof p.text_reader === 'string' ? p.text_reader : '',
  };
}

function same(a: ServedRegionProfile | null, b: ServedRegionProfile | null): boolean {
  if (a === null || b === null) return a === b;
  return (
    a.name === b.name &&
    a.display_name === b.display_name &&
    a.display_name_singular === b.display_name_singular &&
    a.region_class_name === b.region_class_name &&
    a.text_reader === b.text_reader
  );
}

class RegionProfileStore {
  profile = $state<ServedRegionProfile | null>(null);
  loaded = $state<boolean>(false);
  /** Set once a later observation disagrees with the seeded profile. */
  changed = $state<boolean>(false);

  get configured(): boolean {
    return this.loaded && this.profile !== null;
  }

  /** Records the profile the UI is built from. */
  seed(p: ServedRegionProfile | null | undefined): void {
    this.profile = normalize(p);
    this.loaded = true;
    this.changed = false;
  }

  /** A later reading (a `/health` poll, or a region route's 409). */
  observe(p: ServedRegionProfile | null | undefined): void {
    if (!this.loaded || this.changed) return;
    if (same(this.profile, normalize(p))) return;
    this.changed = true;
    toastStore.push({ kind: 'warn', text: REGION_PROFILE_CHANGED_NOTICE, ttl_ms: 0 });
  }

  /** Test-only. */
  reset(): void {
    this.profile = null;
    this.loaded = false;
    this.changed = false;
  }
}

export const regionProfileStore = new RegionProfileStore();

setRegionProfileUnavailableListener(() => regionProfileStore.observe(null));

let inflight: Promise<ServedRegionProfile | null> | null = null;

/**
 * Boot entry point: resolves the served region profile, seeds the store
 * and installs the region slot. Never throws; memoized so client-side
 * navigations (which re-run `load()`) don't refetch.
 */
export function loadRegionProfile(
  fetchHealth: typeof getHealth = getHealth,
): Promise<ServedRegionProfile | null> {
  if (regionProfileStore.loaded) return Promise.resolve(regionProfileStore.profile);
  if (inflight) return inflight;
  inflight = (async () => {
    let profile: ServedRegionProfile | null = null;
    try {
      const h = await fetchHealth(AbortSignal.timeout(REGION_PROFILE_TIMEOUT_MS));
      profile = normalize(h?.region_profile);
    } catch {
      profile = null;
    }
    regionProfileStore.seed(profile);
    installServedRegionProfile(profile);
    inflight = null;
    return profile;
  })();
  return inflight;
}
