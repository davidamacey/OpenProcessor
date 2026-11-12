import { describe, expect, it } from 'vitest';
import { ItemFilterState } from './itemFilterState.svelte';

const BOX: [number, number, number, number] = [-1.5, 0, 2, 3.25];

describe('ItemFilterState viz_box', () => {
  it('spells the box in the query, the body and the URL, and reads it back', () => {
    const s = new ItemFilterState();
    s.vizBox = BOX;
    expect(s.toQuery()).toMatchObject({ viz_box: '-1.5,0,2,3.25' });
    expect(s.toBody()).toMatchObject({ viz_box: BOX });
    const params = new URLSearchParams();
    s.toUrl(params);
    expect(params.get('viz_box')).toBe('-1.5,0,2,3.25');
    const back = new ItemFilterState();
    back.fromUrl(params);
    expect(back.vizBox).toEqual(BOX);
  });

  it('drops an unparsable or inverted box', () => {
    for (const raw of ['1,2', 'a,b,c,d', '2,0,1,1', '0,2,1,1']) {
      const s = new ItemFilterState();
      s.fromUrl(new URLSearchParams({ viz_box: raw }));
      expect(s.vizBox).toBeNull();
    }
  });

  it('is one chip that clears the box, and clear() resets it', () => {
    const s = new ItemFilterState();
    s.vizBox = BOX;
    const chip = s.chips().find((c) => c.param === 'viz_box');
    expect(chip?.label).toBe('Area: embedding box');
    chip?.clear();
    expect(s.vizBox).toBeNull();
    s.vizBox = BOX;
    s.clear();
    expect(s.vizBox).toBeNull();
    expect(s.isEmpty).toBe(true);
  });
});
