/**
 * `profilesAvailability`: the "not yet deployed" gate for region profiles
 * (CRUD). They serve no capability signal (the
 * `detection_profile` axis's `settable` is per `/methods` entry, and an
 * unconfigured project serves an empty axis), so this probes
 * `GET {prefix}/region_profiles` once per project (404/501 = every profile
 * surface absent). See `ConfigAvailability`. Reset
 * on a project switch.
 */
import { listRegionProfiles } from '$lib/api';
import { ConfigAvailability } from '$lib/config/configAvailability.svelte';
import { onProjectChange } from '$lib/projectChange';

export const profilesAvailability = new ConfigAvailability(() => listRegionProfiles());

onProjectChange(() => profilesAvailability.reset());
