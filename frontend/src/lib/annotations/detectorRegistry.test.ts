import { describe, it, expect } from 'vitest';
import { paletteForRole, isMutedTag, PALETTES } from './detectorRegistry';

/**
 * W0 naming-sweep finding m9: chip color is now keyed off a served
 * vocabulary entry's `role`, not the detector id. This is the equivalence
 * proof that every documented role maps to a distinct, correct palette,
 * and that an unknown/absent role degrades to the neutral chip.
 */
describe('paletteForRole', () => {
  it.each([
    ['detector', PALETTES.blue],
    ['segmenter', PALETTES.purple],
    ['ocr', PALETTES.amber],
    ['verifier', PALETTES.teal],
    ['human', PALETTES.emerald],
    ['classifier', PALETTES.indigo],
    ['proposal', PALETTES.sky],
  ] as const)('maps role %s to its palette', (role, expected) => {
    expect(paletteForRole(role)).toEqual(expected);
  });

  it('falls back to the neutral zinc palette for an unrecognized role', () => {
    expect(paletteForRole('some_future_role')).toEqual(PALETTES.zinc);
  });

  it('falls back to zinc for a null/undefined role (id not in the vocabulary)', () => {
    expect(paletteForRole(null)).toEqual(PALETTES.zinc);
    expect(paletteForRole(undefined)).toEqual(PALETTES.zinc);
  });
});

describe('isMutedTag', () => {
  const config = { mutedTagPattern: /miss|reject/ };

  it('matches a tag against the pattern', () => {
    expect(isMutedTag(config, 'miss')).toBe(true);
    expect(isMutedTag(config, 'hit')).toBe(false);
  });

  it('treats a null/absent tag as never muted', () => {
    expect(isMutedTag(config, null)).toBe(false);
  });
});
