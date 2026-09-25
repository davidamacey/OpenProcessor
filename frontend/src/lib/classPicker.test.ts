import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import {
  itemHintClassIds,
  quickAssignClasses,
  resolveConfirmClassId,
  searchClasses,
} from './classPicker';
import { isItemClassTarget } from '$lib/classVisibility';
import {
  installServedRegionProfile,
  registeredSlots,
  resetDeploymentSlots,
} from '$lib/annotations/registeredSlots';
import { WIDGET_TAG_PROFILE } from '$lib/test/fixtures/regionSlot';
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

// #36 item 1 (X2/R1): GET {API_PREFIX}/classes now serves kind directly —
// it must win over the slot registry, both for excluding a region class
// (even one the local registry doesn't know about) and for NOT excluding
// a class the registry happens to bind but the backend now says is 'item'.
describe('item-class targets: served kind (#36 item 1)', () => {
  it('excludes a class the backend marks kind: region, with no slot registered at all', () => {
    resetDeploymentSlots();
    installServedRegionProfile(null);
    try {
      expect(isItemClassTarget({ name: 'some_region_class', kind: 'region' })).toBe(
        false,
      );
    } finally {
      resetDeploymentSlots();
    }
  });

  it('keeps a class the backend marks kind: item, even if a slot happens to bind that name', () => {
    installServedRegionProfile(WIDGET_TAG_PROFILE);
    try {
      expect(
        isItemClassTarget({
          name: WIDGET_TAG_PROFILE.region_class_name,
          kind: 'item',
        }),
      ).toBe(true);
    } finally {
      resetDeploymentSlots();
    }
  });

  it('falls back to the slot registry when kind is absent (an older backend)', () => {
    installServedRegionProfile(WIDGET_TAG_PROFILE);
    try {
      expect(isItemClassTarget({ name: WIDGET_TAG_PROFILE.region_class_name })).toBe(
        false,
      );
      expect(isItemClassTarget({ name: 'miata' })).toBe(true);
    } finally {
      resetDeploymentSlots();
    }
  });
});

describe('item-class targets with no region profile', () => {
  it('excludes no class: the region class is only special when the backend serves a profile for it', () => {
    resetDeploymentSlots();
    installServedRegionProfile(null);
    try {
      expect(isItemClassTarget({ name: WIDGET_TAG_PROFILE.region_class_name })).toBe(
        true,
      );
    } finally {
      resetDeploymentSlots();
    }
  });
});

// R1 (docs/design/visual-audit-2026-09-24.md): the slot-bound region class
// topped the picker and quick-assign row (most validated), so `/` + Enter
// labeled an item as a region. Resolved through the live slot registry,
// never a hardcoded name.
describe('item-class targets (visual audit R1)', () => {
  // The served region profile binds the region slot to its class.
  beforeAll(() => installServedRegionProfile(WIDGET_TAG_PROFILE));
  afterAll(() => resetDeploymentSlots());
  const slotClassName = WIDGET_TAG_PROFILE.region_class_name;

  function poolWithSlotClass(): RegistryClass[] {
    return [
      cls({ id: 80, name: slotClassName, validated_count: 163 }),
      cls({ id: 1, name: 'miata', validated_count: 35 }),
      cls({ id: 2, name: 'touringbike', validated_count: 0 }),
      cls({ id: 3, name: 'class_b', validated_count: 3 }),
    ];
  }

  it('the registry binds at least one slot to a class (precondition)', () => {
    expect(registeredSlots.some((s) => s.bind.className === slotClassName)).toBe(true);
    expect(isItemClassTarget({ name: slotClassName })).toBe(false);
    expect(isItemClassTarget({ name: 'miata' })).toBe(true);
  });

  it('quickAssignClasses never offers the slot-bound class, however validated', () => {
    const row = quickAssignClasses(poolWithSlotClass(), [], 10);
    expect(row.map((c) => c.name)).not.toContain(slotClassName);
    expect(row[0].name).toBe('miata');
  });

  it("quickAssignClasses puts the item's own hinted classes first, in hint order", () => {
    const row = quickAssignClasses(poolWithSlotClass(), [2, null, 3], 10);
    expect(row.map((c) => c.id)).toEqual([2, 3, 1]);
  });

  it('quickAssignClasses ignores a hint pointing at the slot-bound class', () => {
    const row = quickAssignClasses(poolWithSlotClass(), [80], 10);
    expect(row.map((c) => c.id)).toEqual([1, 3, 2]);
  });

  it('searchClasses with an empty query ranks preferred ids first', () => {
    const out = searchClasses(poolWithSlotClass(), '', undefined, [3]);
    expect(out[0].id).toBe(3);
  });

  it('searchClasses ignores preferred ids once the operator types a query', () => {
    const out = searchClasses(poolWithSlotClass(), 'mia', undefined, [3]);
    expect(out.map((c) => c.id)).toEqual([1]);
  });

  it('itemHintClassIds reads proposal, current, VLM and model ids in that order', () => {
    expect(
      itemHintClassIds({
        proposed_class_id: 5,
        class_id: null,
        vlm_suggested_class_id: 7,
        probe_pred_class_id: 9,
      }),
    ).toEqual([5, 7, 9]);
    expect(itemHintClassIds(null)).toEqual([]);
  });
});
