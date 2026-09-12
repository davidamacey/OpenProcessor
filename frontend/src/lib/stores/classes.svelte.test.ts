import { afterEach, describe, expect, it } from 'vitest';
import { classesStore } from './classes.svelte';
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

describe('classesStore.topNForCluster — hides license_plate but stays unfiltered elsewhere', () => {
  afterEach(() => {
    classesStore.classes = [];
  });

  it('never returns license_plate even when its validated_count would rank it first', () => {
    classesStore.classes = [
      cls({ id: 80, name: 'license_plate', validated_count: 999_999 }),
      cls({ id: 8, name: 'bmw', validated_count: 6 }),
      cls({ id: 4, name: 'audi', validated_count: 2 }),
    ];
    const top = classesStore.topNForCluster(0, 10);
    expect(top.map((c) => c.name)).not.toContain('license_plate');
    expect(top[0]?.name).toBe('bmw');
  });

  it('a direct by-id lookup still resolves license_plate — the store itself stays unfiltered', () => {
    classesStore.classes = [cls({ id: 80, name: 'license_plate', validated_count: 999_999 })];
    expect(classesStore.byId(80)?.name).toBe('license_plate');
    expect(classesStore.byName('license_plate')?.id).toBe(80);
  });
});
