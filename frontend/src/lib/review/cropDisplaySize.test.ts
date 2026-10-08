import { describe, expect, it } from 'vitest';
import { CROP_UPSCALE_CAP, capCropDisplayStyle } from './cropDisplaySize';

describe('capCropDisplayStyle (DQ-M5)', () => {
  it('caps a tiny crop (98x106) to the upscale factor instead of filling the panel', () => {
    const style = capCropDisplayStyle(98, 106);
    expect(style).toContain(`max-width:min(100%, ${98 * CROP_UPSCALE_CAP}px)`);
    expect(style).toContain(`max-height:min(100%, ${106 * CROP_UPSCALE_CAP}px)`);
  });

  it('never caps below the natural size (capFactor never less than 1x)', () => {
    const style = capCropDisplayStyle(400, 300, 1);
    expect(style).toContain('max-width:min(100%, 400px)');
    expect(style).toContain('max-height:min(100%, 300px)');
  });

  it('still lets a large crop shrink to fit — the cap is a ceiling, not a floor', () => {
    // A 2000px-wide crop capped at 4x (8000px) is still bounded by the
    // 100% container width via the min(), so it shrinks normally.
    const style = capCropDisplayStyle(2000, 1500);
    expect(style).toContain('max-width:min(100%, 8000px)');
  });

  it('returns empty (no inline cap) before natural size is known', () => {
    expect(capCropDisplayStyle(0, 0)).toBe('');
    expect(capCropDisplayStyle(NaN, 100)).toBe('');
  });

  it('respects a custom capFactor', () => {
    const style = capCropDisplayStyle(100, 50, 3);
    expect(style).toContain('max-width:min(100%, 300px)');
    expect(style).toContain('max-height:min(100%, 150px)');
  });
});
