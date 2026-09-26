/**
 * F8 D1 (OpenProcessor d817605): how `/review` presents the probe's
 * opinion on an item, from the served `probe_in_scope` / `probe_disagreement`.
 *
 * - `probe_in_scope === false`: the item's class is outside the probe's
 *   classes. The probe has no opinion, so its out-of-vocabulary top-1 is
 *   not shown as a prediction, and never as agreement.
 * - OpenProcessor main 9e217f0 adds `probe_actionable` — server-computed
 *   from in-scope + disagreement + the server's own confidence threshold
 *   (`OP_PROBE_ACTIONABLE_MIN_CONFIDENCE`). "Accept model's class" is
 *   offered ONLY when `probe_actionable === true`. A served disagreement
 *   that isn't actionable (below the server's confidence floor) renders
 *   as a muted "model unsure: <predicted class>" with no Accept button —
 *   distinct from a plain agreeing prediction. No client-side threshold
 *   anywhere; the frontend only ever branches on the served booleans.
 */
import type { Crop } from '$lib/types';

export type ProbeOpinionKind = 'no_opinion' | 'prediction' | 'unsure' | 'none';

export interface ProbeOpinion {
  kind: ProbeOpinionKind;
  showAccept: boolean;
}

export const NO_OPINION_TEXT = "no opinion (outside the probe's classes)";

export function probeOpinion(
  item: Pick<
    Crop,
    'class_id' | 'probe_in_scope' | 'probe_disagreement' | 'probe_actionable'
  > & {
    probe_pred_class?: string | null;
    probe_pred_class_id?: number | null;
  },
): ProbeOpinion {
  if (item.probe_in_scope === false) return { kind: 'no_opinion', showAccept: false };
  if (!item.probe_pred_class) return { kind: 'none', showAccept: false };
  if (item.probe_disagreement === true && item.probe_actionable !== true) {
    return { kind: 'unsure', showAccept: false };
  }
  const showAccept =
    item.probe_disagreement === true &&
    item.probe_actionable === true &&
    item.probe_pred_class_id != null &&
    item.probe_pred_class_id !== item.class_id;
  return { kind: 'prediction', showAccept };
}
