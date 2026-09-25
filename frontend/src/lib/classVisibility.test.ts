import { describe, expect, it } from 'vitest';
import { isAssignableClass, isPickerHiddenClass } from './classVisibility';

describe('isAssignableClass', () => {
  it('accepts widget_tag — reverted 2026-09-12, must stay a normal assignable class', () => {
    expect(isAssignableClass({ name: 'widget_tag' })).toBe(true);
    expect(isAssignableClass({ name: 'WIDGET_TAG' })).toBe(true);
  });

  it('rejects deprecated classes', () => {
    expect(isAssignableClass({ name: 'acura', deprecated: true })).toBe(false);
  });

  it('accepts an ordinary active class', () => {
    expect(isAssignableClass({ name: 'widget_a', deprecated: false })).toBe(true);
  });
});

describe('isPickerHiddenClass', () => {
  it('hides nothing today — kept as a named chokepoint for a future per-class toggle', () => {
    expect(isPickerHiddenClass('widget_tag')).toBe(false);
    expect(isPickerHiddenClass('widget_a')).toBe(false);
  });
});
