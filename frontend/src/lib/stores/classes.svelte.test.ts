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

describe('classesStore.topNForCluster — license_plate is a normal class', () => {
  afterEach(() => {
    classesStore.classes = [];
  });

  // Hiding license_plate here was reverted 2026-09-12 — it ranks like any
  // other class now.
  it('ranks license_plate by validated_count like any other class', () => {
    classesStore.classes = [
      cls({ id: 80, name: 'license_plate', validated_count: 999_999 }),
      cls({ id: 8, name: 'bmw', validated_count: 6 }),
      cls({ id: 4, name: 'audi', validated_count: 2 }),
    ];
    const top = classesStore.topNForCluster(0, 10);
    expect(top[0]?.name).toBe('license_plate');
  });

  it('a direct by-id lookup resolves license_plate', () => {
    classesStore.classes = [
      cls({ id: 80, name: 'license_plate', validated_count: 999_999 }),
    ];
    expect(classesStore.byId(80)?.name).toBe('license_plate');
    expect(classesStore.byName('license_plate')?.id).toBe(80);
  });
});
