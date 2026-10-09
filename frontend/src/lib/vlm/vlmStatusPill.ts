/**
 * How `/models` renders a VLM row's served `status` (question A-9: the row
 * carries the endpoint's own status, `ready` / `unprobed` / `probe_failed`
 * / `unreachable`, besides the Triton statuses). The served
 * `VlmEndpointList.labels.status` names it when the id matches; an id with
 * no served label prints raw. The Triton statuses keep their existing
 * pill. Nothing here decides what a status means.
 */
import { modelStatusPill, type StatusPill } from '$lib/modelStatus';
import type { ModelInfo } from '$lib/types';

const NEUTRAL = 'bg-zinc-700/40 text-zinc-400 border-zinc-600';
const TRITON_STATUSES = new Set([
  'ready',
  'not_ready',
  'unavailable',
  'not_configured',
  'not_installed',
]);

export function vlmRowStatusPill(
  model: Pick<ModelInfo, 'status' | 'optional'>,
  labels: Record<string, string> | null,
): StatusPill {
  const status = String(model.status);
  const served = labels?.[status];
  if (TRITON_STATUSES.has(status)) {
    const pill = modelStatusPill(model);
    return served ? { ...pill, label: served } : pill;
  }
  return { label: served ?? status, className: NEUTRAL, title: null };
}
