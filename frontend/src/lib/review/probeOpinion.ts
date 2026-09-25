/**
 * F8 D1: how `/review` presents the probe's opinion on an item, from the
 * served `probe_in_scope` / `probe_disagreement` (OpenProcessor d817605).
 *
 * - `probe_in_scope === false`: the item's class is outside the probe's
 *   classes. The probe has no opinion, so its out-of-vocabulary top-1 is
 *   not shown as a prediction, and never as agreement.
 * - "Accept model's class" only when the server says the probe DISAGREES
 *   (`probe_disagreement === true`) and serves a class id. A null
 *   disagreement (not scored, or no opinion) never offers it.
 */
import type { Crop } from '$lib/types';

export type ProbeOpinionKind = 'no_opinion' | 'prediction' | 'none';

export interface ProbeOpinion {
  kind: ProbeOpinionKind;
  showAccept: boolean;
}

export const NO_OPINION_TEXT = "no opinion (outside the probe's classes)";

export function probeOpinion(
  item: Pick<Crop, 'class_id' | 'probe_in_scope' | 'probe_disagreement'> & {
    probe_pred_class?: string | null;
    probe_pred_class_id?: number | null;
  },
): ProbeOpinion {
  if (item.probe_in_scope === false) return { kind: 'no_opinion', showAccept: false };
  if (!item.probe_pred_class) return { kind: 'none', showAccept: false };
  const showAccept =
    item.probe_disagreement === true &&
    item.probe_pred_class_id != null &&
    item.probe_pred_class_id !== item.class_id;
  return { kind: 'prediction', showAccept };
}
