/**
 * `combineAvailability` — the "not yet deployed" gate for OpenProcessor
 * P4 (combine projects).
 *
 * This is NOT backward compatibility. The backend serves no capability
 * flag and no job list (plan question P4-1), so the one way to know the
 * combine router is mounted is to ask it about a job that cannot exist:
 * a mounted route answers 404 with its own structured
 * `detail.error === 'combine_not_found'`; a backend without the router
 * answers a plain 404 (or 501). The first is "available", the second is
 * "absent", anything else is an error that leaves `available` unknown.
 *
 * Global (combine is not project-scoped), so it is NOT reset on a
 * project switch. When the backend serves a real signal, replace this
 * probe with it rather than keeping both.
 */
import { getCombineJob, isCombineNotFound } from '$lib/api_combine';
import { ConfigAvailability } from '$lib/config/configAvailability.svelte';

/** The id of a job that cannot exist. */
export const COMBINE_PROBE_JOB_ID = '__probe__';

export async function probeCombine(): Promise<void> {
  try {
    await getCombineJob(COMBINE_PROBE_JOB_ID);
  } catch (e) {
    if (isCombineNotFound(e)) return;
    throw e;
  }
}

export const combineAvailability = new ConfigAvailability(probeCombine);
