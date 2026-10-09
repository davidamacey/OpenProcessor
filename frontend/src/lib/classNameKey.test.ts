import { describe, expect, it } from 'vitest';
import { findActiveClassByName, normalizeClassName } from './classNameKey';

const c = (id: number, name: string, deprecated = false) => ({ id, name, deprecated });

describe('normalizeClassName', () => {
  // The backend's own cases (src/utils/class_names.py normalize_class_name):
  // lowercase, every RUN of non-alphanumerics becomes one `_`, edges trimmed.
  it.each([
    ['traffic light', 'traffic_light'],
    ['  Hot  Dog ', 'hot_dog'],
    ['t-shirt', 't_shirt'],
    ['widget', 'widget'],
    ['__x__', 'x'],
    ['a - b', 'a_b'],
    ['', ''],
  ])('matches the backend rule: %j -> %j', (raw, slug) => {
    expect(normalizeClassName(raw)).toBe(slug);
  });
  it('folds case, spaces and hyphens to an underscore', () => {
    expect(normalizeClassName('Blue Widget')).toBe('blue_widget');
    expect(normalizeClassName('blue-widget')).toBe('blue_widget');
    expect(normalizeClassName('BLUE_WIDGET')).toBe('blue_widget');
  });
});

describe('findActiveClassByName', () => {
  it('matches across the folded spellings', () => {
    expect(findActiveClassByName([c(1, 'blue_widget')], 'Blue Widget')?.id).toBe(1);
    expect(findActiveClassByName([c(1, 'Blue-Widget')], 'blue widget')?.id).toBe(1);
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
