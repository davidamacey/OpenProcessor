import { describe, expect, it } from 'vitest';
import {
  dominantShareText,
  dominantShareTitle,
  cohesionText,
  COHESION_TOOLTIP,
} from './clusterCardText';

describe('cluster purity display text (visual audit C1/K4)', () => {
  it('dominant share reads the label share, labelled as such', () => {
    expect(dominantShareText({ dominant_pct: 1 })).toBe('100% of labeled');
    expect(dominantShareText({ dominant_pct: null })).toBeNull();
  });

  it('F-37: the served geometry purity reads as cohesion with its n', () => {
    expect(cohesionText({ purity: 0.19, purity_n: 984 })).toBe('cohesion 19% · n=984');
    expect(cohesionText({ purity: 0.19, purity_n: null })).toBe('cohesion 19%');
    expect(cohesionText({ purity: null, purity_n: 5 })).toBeNull();
    expect(COHESION_TOOLTIP).toContain(
      'share of measured members whose nearest cluster centre is this one',
    );
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
