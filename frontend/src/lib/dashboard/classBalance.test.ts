import { describe, expect, it } from 'vitest';
import { sortClassBalance, type ClassBalanceRow } from './classBalance';

function row(over: Partial<ClassBalanceRow>): ClassBalanceRow {
  return { class_id: 1, class_name: 'z', count: 0, validated_count: 0, ...over };
}

describe('sortClassBalance (DQ-m11)', () => {
  it('sorts by validated_count desc when counts differ', () => {
    const rows = [
      row({ class_name: 'a', validated_count: 1 }),
      row({ class_name: 'b', validated_count: 5 }),
    ];
    expect(sortClassBalance(rows).map((r) => r.class_name)).toEqual(['b', 'a']);
  });

  it('reproduces the DQ-m11 repro: all validated_count=0 breaks ties by total count, not alphabetically', () => {
    // Alphabetically, 'bin' and 'crate' would sort before 'pallet'/'tote' —
    // the bug hid pallet/tote (557/large counts) behind small alphabetically-
    // earlier classes once every validated_count tied at 0.
    const rows = [
      row({ class_name: 'crate', count: 2, validated_count: 0 }),
      row({ class_name: 'bin', count: 3, validated_count: 0 }),
      row({ class_name: 'pallet', count: 557, validated_count: 0 }),
      row({ class_name: 'tote', count: 400, validated_count: 0 }),
    ];
    const sorted = sortClassBalance(rows).map((r) => r.class_name);
    expect(sorted).toEqual(['pallet', 'tote', 'bin', 'crate']);
  });

  it('falls back to class_name alphabetically only when both validated_count and count tie', () => {
    const rows = [
      row({ class_name: 'zebra', count: 5, validated_count: 0 }),
      row({ class_name: 'apple', count: 5, validated_count: 0 }),
    ];
    expect(sortClassBalance(rows).map((r) => r.class_name)).toEqual(['apple', 'zebra']);
  });

  it('does not mutate the input array', () => {
    const rows = [
      row({ class_name: 'b', validated_count: 1 }),
      row({ class_name: 'a', validated_count: 5 }),
    ];
    const original = [...rows];
    sortClassBalance(rows);
    expect(rows).toEqual(original);
  });
});
