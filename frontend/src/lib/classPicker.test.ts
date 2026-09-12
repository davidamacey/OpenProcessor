import { describe, expect, it } from 'vitest';
import { resolveConfirmClassId, searchClasses } from './classPicker';
import type { OpClass } from '$lib/types';

function cls(over: Partial<OpClass> & { id: number; name: string }): OpClass {
  return {
    group: null,
    count: 0,
    validated_count: 0,
    cluster_size: 0,
    added_at: '2026-01-01T00:00:00Z',
    ...over,
  };
}

// A pool shaped like the audit finding: 84 classes, only a handful with any
// validated_count, `scion` among the zero-sample ones that can never climb
// into topNForCluster(0, 10) on its own.
function bigPool(): OpClass[] {
  const out: OpClass[] = [];
  for (let i = 0; i < 80; i++) {
    out.push(cls({ id: i, name: `class_${i}`, validated_count: 80 - i }));
  }
  out.push(cls({ id: 900, name: 'scion', validated_count: 0 }));
  out.push(cls({ id: 901, name: 'acura', validated_count: 0 }));
  out.push(cls({ id: 902, name: 'brand_b', validated_count: 0 }));
  out.push(
    cls({
      id: 903,
      name: 'zz_audit_test_class_renamed',
      validated_count: 0,
      deprecated: true,
    }),
  );
  return out; // 84 total, 1 deprecated
}

describe('searchClasses', () => {
  it('ranks a prefix match above a substring-only match', () => {
    const pool = [
      cls({ id: 1, name: 'subaru_impreza' }), // 'sci' is not a substring
      cls({ id: 2, name: 'sci' }), // exact
      cls({ id: 3, name: 'scion' }), // prefix
      cls({ id: 4, name: 'old_scion_2000' }), // substring only
    ];
    const results = searchClasses(pool, 'sci');
    expect(results.map((c) => c.name)).toEqual(['sci', 'scion', 'old_scion_2000']);
  });

  it('excludes deprecated classes even when they match the query', () => {
    const pool = bigPool();
    const results = searchClasses(pool, 'zz_audit');
    expect(results).toHaveLength(0);
  });

  it('returns all 84 classes (minus deprecated) for an empty query, not just the top 10', () => {
    const pool = bigPool();
    const results = searchClasses(pool, '');
    expect(pool).toHaveLength(84);
    expect(results).toHaveLength(83); // 84 - 1 deprecated
    // Still ordered most-validated-first, same convention as topNForCluster.
    expect(results[0].name).toBe('class_0');
  });

  it('finds a zero-sample, zero-validated class like "scion" that topNForCluster(0, 10) would never surface', () => {
    const pool = bigPool();
    const results = searchClasses(pool, 'sci');
    expect(results.map((c) => c.name)).toContain('scion');
  });

  it('falls back to subsequence matching for near-misses', () => {
    const pool = [cls({ id: 1, name: 'volkswagen' })];
    expect(searchClasses(pool, 'vlkwgn').map((c) => c.name)).toEqual(['volkswagen']);
    expect(searchClasses(pool, 'zzz')).toHaveLength(0);
  });

  it('respects an explicit limit', () => {
    const pool = bigPool();
    expect(searchClasses(pool, '', 10)).toHaveLength(10);
  });
});

describe('searchClasses — license_plate is hidden from assignment search', () => {
  function poolWithPlate(): OpClass[] {
    return [
      cls({ id: 8, name: 'bmw', validated_count: 6 }),
      cls({ id: 80, name: 'license_plate', validated_count: 72 }),
      cls({ id: 999, name: 'license_plate_holder', validated_count: 0 }),
    ];
  }

  it('excludes license_plate from a "license" search (near-miss license_plate_holder still matches)', () => {
    const results = searchClasses(poolWithPlate(), 'license');
    expect(results.map((c) => c.name)).not.toContain('license_plate');
    expect(results.map((c) => c.name)).toContain('license_plate_holder');
  });

  it('returns zero results for "license_plate" (exact query, no near-miss pool member)', () => {
    const results = searchClasses(
      poolWithPlate().filter((c) => c.name !== 'license_plate_holder'),
      'license_plate',
    );
    expect(results).toHaveLength(0);
  });

  it('excludes license_plate from a "lp" search even though it subsequence-matches', () => {
    // 'lp' is also a subsequence of "license_plate_holder", which must
    // still surface normally — only the exact-name license_plate is hidden.
    const results = searchClasses(poolWithPlate(), 'lp');
    expect(results.map((c) => c.name)).not.toContain('license_plate');
  });

  it('omits license_plate from the default (empty-query) full list', () => {
    const results = searchClasses(poolWithPlate(), '');
    expect(results.map((c) => c.name)).not.toContain('license_plate');
    expect(results.map((c) => c.name)).toContain('license_plate_holder');
  });
});

describe('resolveConfirmClassId', () => {
  it('prefers proposed_class_id over the current class_id', () => {
    expect(resolveConfirmClassId({ proposed_class_id: 5, class_id: 1 })).toBe(5);
  });

  it('falls back to class_id when there is no proposal', () => {
    expect(resolveConfirmClassId({ proposed_class_id: null, class_id: 1 })).toBe(1);
  });

  it('returns null when both are absent — the P1-5 "blank proposal" case', () => {
    expect(resolveConfirmClassId({ proposed_class_id: null, class_id: null })).toBeNull();
  });

  it('returns null for a null/undefined item', () => {
    expect(resolveConfirmClassId(null)).toBeNull();
    expect(resolveConfirmClassId(undefined)).toBeNull();
  });
});
