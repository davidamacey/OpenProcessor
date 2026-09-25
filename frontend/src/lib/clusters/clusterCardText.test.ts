import { describe, expect, it } from 'vitest';
import {
  dominantShareText,
  dominantShareTitle,
  geometryPurityText,
  purityBasisLabel,
} from './clusterCardText';

describe('cluster purity display text (visual audit C1/K4)', () => {
  it('dominant share reads the label share, labelled as such', () => {
    expect(dominantShareText({ dominant_pct: 1 })).toBe('100% of labeled');
    expect(dominantShareText({ dominant_pct: null })).toBeNull();
  });

  it('geometry purity is named as geometry, not a raw basis id', () => {
    expect(geometryPurityText({ purity: 0.03, purity_basis: 'nearest_centroid' })).toBe(
      '3% geometry',
    );
    expect(
      geometryPurityText({ purity: null, purity_basis: 'nearest_centroid' }),
    ).toBeNull();
    expect(purityBasisLabel('some_other_basis')).toBe('some other basis');
  });

  it('the dominant-share tooltip cites the served counts', () => {
    expect(
      dominantShareTitle({
        dominant_count: 616,
        labelled_count: 616,
        size: 616,
        dominant_class_name: 'class_b',
      }),
    ).toBe('616 of 616 labeled members are class_b (cluster size 616)');
  });
});
