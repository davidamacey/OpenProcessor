import { describe, expect, it } from 'vitest';
import type { BoxTone } from '$lib/types';
import { boxToneClass } from './boxTone';

const TONES: BoxTone[] = ['validated', 'machine', 'unlabeled', 'holdout', 'rejected'];

describe('boxToneClass', () => {
  it('one class per tone', () => {
    const classes = TONES.map((t) => boxToneClass(t));
    expect(new Set(classes).size).toBe(TONES.length);
    for (const c of classes) expect(c).toMatch(/^border-/);
  });

  it("gives the detector's undecided proposals the detector tone, and decided ones their own", () => {
    expect(boxToneClass('unlabeled', 'detector')).toBe('border-tone-detector');
    expect(boxToneClass('machine', 'detector')).toBe('border-tone-detector');
    expect(boxToneClass('validated', 'detector')).toBe('border-success');
    expect(boxToneClass('unlabeled', 'class')).toBe('border-fg-3');
  });
});
