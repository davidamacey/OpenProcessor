/**
 * `vlmAvailability`: the "not yet deployed" gate for OpenProcessor W9 (the
 * VLM endpoint registry). W9 serves no capability flag for its registry
 * routes, so this probes the GLOBAL `GET {prefix}/vlm/endpoints` once
 * (404/501 = every registry surface absent, no other `/vlm/*` request
 * fires). Unlike the project-scoped gates it is deployment-wide, so it is
 * NOT registered with the project-change resets. See `ConfigAvailability`.
 *
 * The probe IS the list read, so its served `labels.status` is kept for
 * the `/models` page, which names a VLM row's status with it without a
 * second request.
 */
import { ConfigAvailability } from '$lib/config/configAvailability.svelte';
import { listVlmEndpoints } from '$lib/api_vlm';

class VlmStatusLabels {
  /** The served `labels.status` of the probe's answer, or null. */
  status = $state<Record<string, string> | null>(null);
}

export const vlmStatusLabels = new VlmStatusLabels();

export const vlmAvailability = new ConfigAvailability(async () => {
  const list = await listVlmEndpoints();
  vlmStatusLabels.status = list.labels?.status ?? null;
  return list;
});
