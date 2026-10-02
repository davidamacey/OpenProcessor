/**
 * The served words for a refused per-run VLM call (`label_cluster`,
 * `auto_label/start`): 422 `unknown_vlm` (axis / requested / valid ids),
 * 422 `vlm_external_not_acknowledged`, 409 `vlm_not_configured`, 409
 * `vlm_endpoint_unavailable` and a pairing 422 `validation_failed` (its
 * message and how many issues the served report lists). Anything else is
 * the error's own message.
 */
import { configErrorDetail, unknownStrategyDetail } from '$lib/api';

export function vlmRunErrorText(e: unknown): string {
  const unknown = unknownStrategyDetail(e);
  if (unknown) {
    return `unknown ${unknown.axis.replace('_', ' ')} "${unknown.requested}" — valid: ${unknown.valid_ids.join(', ') || 'none'}.`;
  }
  const d = configErrorDetail(e);
  if (d) {
    const n = (d.report?.errors.length ?? 0) + (d.report?.warnings.length ?? 0);
    return d.error === 'validation_failed' && n > 0
      ? `${d.message} (${n} issue${n === 1 ? '' : 's'})`
      : d.message;
  }
  return (e as Error)?.message ?? String(e);
}
