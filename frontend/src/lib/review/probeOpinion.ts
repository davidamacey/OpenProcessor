/**
 * F8 D1 (OpenProcessor 51b05d7): how `/review` presents the probe's
 * opinion on an item, from the served `probe_in_scope` / `probe_disagreement`.
 *
 * - `probe_in_scope === false`: the item's class is outside the probe's
 *   classes. The probe has no opinion, so its out-of-vocabulary top-1 is
 *   not shown as a prediction, and never as agreement.
 * - OpenProcessor main 8990ede adds `probe_actionable` — server-computed
 *   from in-scope + disagreement + the server's own confidence threshold
 *   (`OP_PROBE_ACTIONABLE_MIN_CONFIDENCE`). "Accept model's class" is
 *   offered ONLY when `probe_actionable === true`. A served disagreement
 *   that isn't actionable (below the server's confidence floor) renders
 *   as a muted "model unsure: <predicted class>" with no Accept button —
 *   distinct from a plain agreeing prediction. No client-side threshold
 *   anywhere; the frontend only ever branches on the served booleans.
 * - class-id-display-audit-2026-09-26: `showAccept` used to also require
 *   `item.probe_pred_class_id !== item.class_id` -- a client-side id
 *   comparison redundant with (and riskier than) the served
 *   `probe_disagreement`/`probe_actionable` flags: if `probe_pred_class_id`
 *   were ever a dense export id rather than a registry id, this comparison
 *   could show/hide Accept on a false read. Dropped; the served booleans
 *   alone decide, per the thin-frontend rule.
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
    item.probe_pred_class_id != null;
  return { kind: 'prediction', showAccept };
}
