/**
 * F8 D1 (OpenProcessor d817605): "Accept model's class" is offered only
 * when the served `probe_disagreement` is true; an out-of-scope item reads
 * "no opinion", never a prediction or agreement.
 */
import { describe, expect, it } from 'vitest';
import { probeOpinion } from './probeOpinion';

const base = {
  class_id: 5,
  probe_pred_class: 'widget_b',
  probe_pred_class_id: 7,
};

describe('probeOpinion', () => {
  it('a served disagreement with a different class offers Accept', () => {
    expect(
      probeOpinion({ ...base, probe_in_scope: true, probe_disagreement: true }),
    ).toEqual({ kind: 'prediction', showAccept: true });
  });

  it('never offers Accept when probe_disagreement is null (not scored / no opinion)', () => {
    expect(
      probeOpinion({ ...base, probe_in_scope: null, probe_disagreement: null })
        .showAccept,
    ).toBe(false);
    expect(probeOpinion({ ...base }).showAccept).toBe(false);
  });

  it('agreement (probe_disagreement false) shows the prediction without Accept', () => {
    expect(
      probeOpinion({ ...base, probe_in_scope: true, probe_disagreement: false }),
    ).toEqual({ kind: 'prediction', showAccept: false });
  });

  it('out of the probe classes is "no opinion", never a prediction', () => {
    expect(
      probeOpinion({ ...base, probe_in_scope: false, probe_disagreement: null }),
    ).toEqual({ kind: 'no_opinion', showAccept: false });
  });

  it('nothing to show without a served prediction', () => {
    expect(
      probeOpinion({
        class_id: 5,
        probe_pred_class: null,
        probe_in_scope: true,
        probe_disagreement: true,
      }),
    ).toEqual({ kind: 'none', showAccept: false });
  });
});
