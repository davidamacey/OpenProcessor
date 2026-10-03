import { describe, expect, it } from 'vitest';
import { findActiveClassByName, normalizeClassName } from './classNameKey';

const c = (id: number, name: string, deprecated = false) => ({ id, name, deprecated });

describe('normalizeClassName', () => {
  it('folds case, spaces and hyphens to an underscore', () => {
    expect(normalizeClassName('Sports Car')).toBe('sports_car');
    expect(normalizeClassName('sports-car')).toBe('sports_car');
    expect(normalizeClassName('SPORTS_CAR')).toBe('sports_car');
  });
});

describe('findActiveClassByName', () => {
  it('matches across the folded spellings', () => {
    expect(findActiveClassByName([c(1, 'sports_car')], 'Sports Car')?.id).toBe(1);
    expect(findActiveClassByName([c(1, 'Sports-Car')], 'sports car')?.id).toBe(1);
  });

  it('an ACTIVE class wins over a deprecated one with the same name', () => {
    const classes = [c(1, 'widget', true), c(2, 'Widget')];
    expect(findActiveClassByName(classes, 'widget')?.id).toBe(2);
    expect(findActiveClassByName([...classes].reverse(), 'widget')?.id).toBe(2);
  });

  it('a deprecated-only match is never returned (never assigned to)', () => {
    expect(findActiveClassByName([c(1, 'widget', true)], 'widget')).toBeNull();
  });

  it('no match is null', () => {
    expect(findActiveClassByName([c(1, 'widget')], 'gadget')).toBeNull();
  });
});
