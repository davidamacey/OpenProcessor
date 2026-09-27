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
 * root layout's `load()`. Only a SUCCESSFUL read seeds: a served
 * `region_profile: null` means "not configured". A timeout or network error does not (F-78: a slow
 * first `/health` used to seed "no profile", the next poll disagreed,
 * and a spurious "reload" toast fired while the region tab was missing).
 * Boot retries with a bounded backoff; if every try fails the store stays
 * `unknown` (still no region route is called — fail closed), and the
 * first successful `/health` poll seeds it instead. The root layout keys
 * the page on `seedVersion`, so a late seed re-renders with the region
 * tab and surfaces in place, no reload needed.
 *
 * Only a successful read that differs from an earlier successful read
 * (or a region route's 409 after a profile was seeded) counts as
 * "changed": the store then shows one "reload to apply" notice rather
 * than hot-swapping tabs.
 */

import { getHealth } from '$lib/api';
import { installServedRegionProfile } from '$lib/annotations/registeredSlots';
import { onProjectChange } from '$lib/projectChange';
import { setRegionProfileUnavailableListener } from '$lib/regionProfileUnavailable';
import type { ServedRegionProfile } from '$lib/types';
import { toastStore } from '$stores/toast.svelte';

export const REGION_PROFILE_TIMEOUT_MS = 2000;
/** Waits before boot retries 2 and 3 (bounded: 3 tries in total). */
export const REGION_PROFILE_RETRY_DELAYS_MS = [250, 750];

export const REGION_PROFILE_CHANGED_NOTICE =
  "The backend's region profile changed — reload the page to apply it.";

/** The served profile's own fields (dropping anything else on the wire),
 *  or `null` when none is configured. */
function normalize(p: ServedRegionProfile | null): ServedRegionProfile | null {
  if (!p) return null;
  return {
    name: p.name,
    display_name: p.display_name,
    display_name_singular: p.display_name_singular,
    region_class_name: p.region_class_name,
    text_reader: p.text_reader,
    reads_text: p.reads_text,
    text_hint_enabled: p.text_hint_enabled,
  };
}

function same(a: ServedRegionProfile | null, b: ServedRegionProfile | null): boolean {
  if (a === null || b === null) return a === b;
  return (
    a.name === b.name &&
    a.display_name === b.display_name &&
    a.display_name_singular === b.display_name_singular &&
    a.region_class_name === b.region_class_name &&
    a.text_reader === b.text_reader &&
    a.reads_text === b.reads_text &&
    a.text_hint_enabled === b.text_hint_enabled
  );
}

class RegionProfileStore {
  profile = $state<ServedRegionProfile | null>(null);
  /** True once a successful read (boot or a later poll) has seeded it. */
  loaded = $state<boolean>(false);
  /** True when boot finished without any successful read: not known yet,
   *  NOT "not configured". The first successful poll seeds it. */
  unknown = $state<boolean>(false);
  /** Set once a later observation disagrees with the seeded profile. */
  changed = $state<boolean>(false);
  /** Bumped on every seed; the root layout keys the page on it so a late
   *  seed re-renders every region surface. */
  seedVersion = $state<number>(0);

  get configured(): boolean {
    return this.loaded && this.profile !== null;
  }

  /** Records the profile the UI is built from (a successful read). */
  seed(p: ServedRegionProfile | null): void {
    this.profile = normalize(p);
    this.loaded = true;
    this.unknown = false;
    this.changed = false;
    this.seedVersion += 1;
  }

  /** Boot gave up without a successful read. */
  markUnknown(): void {
    if (this.loaded) return;
    this.unknown = true;
  }

  /** A later successful reading (a `/health` poll, or a region route's
   *  409). Seeds the store when boot never got one; otherwise a reading
   *  that differs from the seeded one raises the reload notice once. */
  observe(p: ServedRegionProfile | null): void {
    if (!this.loaded) {
      if (!this.unknown) return;
      this.seed(p);
      installServedRegionProfile(this.profile);
      return;
    }
    if (this.changed) return;
    if (same(this.profile, normalize(p))) return;
    this.changed = true;
    toastStore.push({ kind: 'warn', text: REGION_PROFILE_CHANGED_NOTICE, ttl_ms: 0 });
  }

  /**
   * Project switch (review §3.8): the served profile is per project, so
   * a different project's profile is NOT a "change" — the store goes back
   * to unseeded (no region slot installed, no region route called) and
   * the `/p/[project]` layout's `loadRegionProfile()` seeds it for the new
   * project with no reload notice. `seedVersion` keeps counting up so the
   * keyed page re-mounts.
   */
  resetForProjectChange(): void {
    this.profile = null;
    this.loaded = false;
    this.unknown = false;
    this.changed = false;
    this.seedVersion += 1;
  }

  /** Test-only. */
  reset(): void {
    this.profile = null;
    this.loaded = false;
    this.unknown = false;
    this.changed = false;
    this.seedVersion = 0;
  }
}

export const regionProfileStore = new RegionProfileStore();

setRegionProfileUnavailableListener(() => regionProfileStore.observe(null));

let inflight: Promise<ServedRegionProfile | null> | null = null;
/** Bumped on a project switch: a boot load started for the previous
 *  project must never seed the new one. */
let loadGeneration = 0;

onProjectChange(() => {
  loadGeneration += 1;
  inflight = null;
  regionProfileStore.resetForProjectChange();
  installServedRegionProfile(null);
});

function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Boot entry point: resolves the served region profile, seeds the store
 * and installs the region slot. Never throws; memoized so client-side
 * navigations (which re-run `load()`) don't refetch. A timeout or network
 * error is retried (3 tries in total); if none succeeds the store is left
 * `unknown` for the first successful `/health` poll to seed.
 */
export function loadRegionProfile(
  fetchHealth: typeof getHealth = getHealth,
  retryDelaysMs: readonly number[] = REGION_PROFILE_RETRY_DELAYS_MS,
): Promise<ServedRegionProfile | null> {
  if (regionProfileStore.loaded) return Promise.resolve(regionProfileStore.profile);
  if (inflight) return inflight;
  const gen = loadGeneration;
  const run = (async () => {
    for (let attempt = 0; attempt <= retryDelaysMs.length; attempt++) {
      if (attempt > 0) await delay(retryDelaysMs[attempt - 1]!);
      if (gen !== loadGeneration) return null;
      // A health poll may have seeded the store while we waited.
      if (regionProfileStore.loaded) break;
      try {
        const h = await fetchHealth(AbortSignal.timeout(REGION_PROFILE_TIMEOUT_MS));
        if (gen !== loadGeneration) return null;
        regionProfileStore.seed(normalize(h.region_profile));
        installServedRegionProfile(regionProfileStore.profile);
        break;
      } catch {
        // Timeout / network error: unknown, not "no profile". Retry.
      }
    }
    if (gen !== loadGeneration) return null;
    if (!regionProfileStore.loaded) regionProfileStore.markUnknown();
    inflight = null;
    return regionProfileStore.profile;
  })();
  inflight = run;
  return run;
}
