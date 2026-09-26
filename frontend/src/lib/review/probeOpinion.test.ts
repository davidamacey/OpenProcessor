/**
 * F8 D1 (OpenProcessor 51b05d7): "Accept model's class" is offered only
 * when the served `probe_disagreement` is true; an out-of-scope item reads
 * "no opinion", never a prediction or agreement.
 *
 * OpenProcessor main 8990ede adds `probe_actionable` — Accept requires
 * `probe_actionable === true`, not just `probe_disagreement === true`. A
 * disagreement that isn't (yet) actionable renders as "unsure", no Accept.
 *
 * class-id-display-audit-2026-09-26: `showAccept` used to also require a
 * client-side `probe_pred_class_id !== class_id` comparison. That's gone —
 * the served `probe_disagreement`/`probe_actionable` flags decide alone.
 */
import { describe, expect, it } from 'vitest';
import { probeOpinion } from './probeOpinion';

const base = {
  class_id: 5,
  probe_pred_class: 'widget_b',
  probe_pred_class_id: 7,
};

describe('probeOpinion', () => {
  it('actionable disagreement with a different class offers Accept', () => {
    expect(
      probeOpinion({
        ...base,
        probe_in_scope: true,
        probe_disagreement: true,
        probe_actionable: true,
      }),
    ).toEqual({ kind: 'prediction', showAccept: true });
  });

  it('disagreeing but not actionable is "unsure", never Accept', () => {
    expect(
      probeOpinion({
        ...base,
        probe_in_scope: true,
        probe_disagreement: true,
        probe_actionable: false,
      }),
    ).toEqual({ kind: 'unsure', showAccept: false });
  });

  it('disagreeing with a null probe_actionable (older/un-backfilled item) is also "unsure"', () => {
    expect(
      probeOpinion({
        ...base,
        probe_in_scope: true,
        probe_disagreement: true,
        probe_actionable: null,
      }),
    ).toEqual({ kind: 'unsure', showAccept: false });
  });

  it('never offers Accept when probe_disagreement is null (not scored / no opinion)', () => {
    expect(
      probeOpinion({
        ...base,
        probe_in_scope: null,
        probe_disagreement: null,
        probe_actionable: null,
      }).showAccept,
    ).toBe(false);
    expect(probeOpinion({ ...base }).showAccept).toBe(false);
  });

  it('agreement (probe_disagreement false) shows the prediction without Accept, even if probe_actionable is true', () => {
    expect(
      probeOpinion({
        ...base,
        probe_in_scope: true,
        probe_disagreement: false,
        probe_actionable: true,
      }),
    ).toEqual({ kind: 'prediction', showAccept: false });
  });

  it('out of the probe classes is "no opinion", never a prediction', () => {
    expect(
      probeOpinion({
        ...base,
        probe_in_scope: false,
        probe_disagreement: null,
        probe_actionable: null,
      }),
    ).toEqual({ kind: 'no_opinion', showAccept: false });
  });

  it('actionable disagreement offers Accept even when probe_pred_class_id equals class_id (served flags decide, not a client id compare)', () => {
    expect(
      probeOpinion({
        ...base,
        probe_pred_class_id: base.class_id,
        probe_in_scope: true,
        probe_disagreement: true,
        probe_actionable: true,
      }),
    ).toEqual({ kind: 'prediction', showAccept: true });
  });

  it('never offers Accept when the served probe_pred_class_id is missing, even if actionable', () => {
    expect(
      probeOpinion({
        ...base,
        probe_pred_class_id: null,
        probe_in_scope: true,
        probe_disagreement: true,
        probe_actionable: true,
      }),
    ).toEqual({ kind: 'prediction', showAccept: false });
  });

  it('nothing to show without a served prediction', () => {
    expect(
      probeOpinion({
        class_id: 5,
        probe_pred_class: null,
        probe_in_scope: true,
        probe_disagreement: true,
        probe_actionable: true,
      }),
    ).toEqual({ kind: 'none', showAccept: false });
  });
});
