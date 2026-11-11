import { describe, expect, it } from 'vitest';
import type { BoxTone } from '$lib/types';
import { boxToneClass } from './boxTone';

const TONES: BoxTone[] = ['validated', 'machine', 'unlabeled', 'holdout', 'rejected'];

describe('boxToneClass', () => {
  it('one class per tone', () => {
    const classes = TONES.map(boxToneClass);
    expect(new Set(classes).size).toBe(TONES.length);
    for (const c of classes) expect(c).toMatch(/^border-/);
  });
});
