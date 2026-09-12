import { describe, expect, it } from 'vitest';
import { isAssignableClass, isPickerHiddenClass } from './classVisibility';

describe('isAssignableClass', () => {
  it('accepts license_plate — reverted 2026-09-12, must stay a normal assignable class', () => {
    expect(isAssignableClass({ name: 'license_plate' })).toBe(true);
    expect(isAssignableClass({ name: 'LICENSE_PLATE' })).toBe(true);
  });

  it('rejects deprecated classes', () => {
    expect(isAssignableClass({ name: 'acura', deprecated: true })).toBe(false);
  });

  it('accepts an ordinary active class', () => {
    expect(isAssignableClass({ name: 'bmw', deprecated: false })).toBe(true);
  });
});

describe('isPickerHiddenClass', () => {
  it('hides nothing today — kept as a named chokepoint for a future per-class toggle', () => {
    expect(isPickerHiddenClass('license_plate')).toBe(false);
    expect(isPickerHiddenClass('bmw')).toBe(false);
  });
});
