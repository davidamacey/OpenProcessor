import { describe, expect, it } from 'vitest';
import { isAssignableClass, isPickerHiddenClass } from './classVisibility';

describe('isAssignableClass', () => {
  it('rejects license_plate (exact, case-insensitive)', () => {
    expect(isAssignableClass({ name: 'license_plate' })).toBe(false);
    expect(isAssignableClass({ name: 'LICENSE_PLATE' })).toBe(false);
    expect(isAssignableClass({ name: 'License_Plate' })).toBe(false);
  });

  it('rejects deprecated classes', () => {
    expect(isAssignableClass({ name: 'acura', deprecated: true })).toBe(false);
  });

  it('does not use substring matching — a near-miss name is still assignable', () => {
    expect(isAssignableClass({ name: 'license_plate_holder' })).toBe(true);
  });

  it('accepts an ordinary active class', () => {
    expect(isAssignableClass({ name: 'bmw', deprecated: false })).toBe(true);
  });
});

describe('isPickerHiddenClass', () => {
  it('is true only for license_plate, exact match', () => {
    expect(isPickerHiddenClass('license_plate')).toBe(true);
    expect(isPickerHiddenClass('LICENSE_PLATE')).toBe(true);
    expect(isPickerHiddenClass('license_plate_holder')).toBe(false);
    expect(isPickerHiddenClass('bmw')).toBe(false);
  });
});
