/**
 * `profilesAvailability`: the "not yet deployed" gate for OpenProcessor W4
 * (region-profile CRUD). W4 serves no capability signal (the
 * `detection_profile` axis's `settable` is per `/methods` entry, and an
 * unconfigured project serves an empty axis), so this probes
 * `GET {prefix}/region_profiles` once per project (404/501 = every profile
 * surface absent). See `ConfigAvailability` and
 * docs/design/w4-profile-editor-ui-plan-2026-09-27.md §0.3 (W4-Q1). Reset
 * on a project switch.
 */
import { listRegionProfiles } from '$lib/api';
import { ConfigAvailability } from '$lib/config/configAvailability.svelte';
import { onProjectChange } from '$lib/projectChange';

export const profilesAvailability = new ConfigAvailability(() => listRegionProfiles());

onProjectChange(() => profilesAvailability.reset());
