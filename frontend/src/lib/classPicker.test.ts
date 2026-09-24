import { describe, expect, it } from 'vitest';
import { resolveConfirmClassId, searchClasses } from './classPicker';
import type { RegistryClass } from '$lib/types';

function cls(over: Partial<RegistryClass> & { id: number; name: string }): RegistryClass {
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
function bigPool(): RegistryClass[] {
  const out: RegistryClass[] = [];
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

describe('searchClasses — widget_tag is a normal assignable class', () => {
  // Hiding a region-bound class from assignment search was reverted
  // 2026-09-12: it stays reachable like any other class.
  it('finds widget_tag like any other class', () => {
    const pool: RegistryClass[] = [
      cls({ id: 8, name: 'bmw', validated_count: 6 }),
      cls({ id: 80, name: 'widget_tag', validated_count: 72 }),
    ];
    const results = searchClasses(pool, 'widget_tag');
    expect(results.map((c) => c.name)).toContain('widget_tag');
  });
});

// The backend (review.py) already resolves proposed_class_id: the VLM's
// proposal, else the item's class, or null when nothing is confirmable.
// The frontend uses it as-is and never second-guesses it.
describe('resolveConfirmClassId', () => {
  it("returns the backend's proposed_class_id", () => {
    expect(resolveConfirmClassId({ proposed_class_id: 5 })).toBe(5);
  });

  it('returns null when the backend says nothing is confirmable', () => {
    expect(resolveConfirmClassId({ proposed_class_id: null })).toBeNull();
    expect(resolveConfirmClassId(null)).toBeNull();
    expect(resolveConfirmClassId(undefined)).toBeNull();
  });
});
