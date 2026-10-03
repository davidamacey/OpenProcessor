import { afterEach, describe, expect, it, vi } from 'vitest';
import { classesStore } from './classes.svelte';
import type { RegistryClass } from '$lib/types';

const ok = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

function cls(over: Partial<RegistryClass> & { id: number; name: string }): RegistryClass {
  return {
    group: null,
    count: 0,
    validated_count: 0,
    cluster_size: 0,
    added_at: '2026-01-01T00:00:00Z',
    adequacy: 'block',
    kind: 'item',
    trainable: 0,
    trainable_gap: 0,
    ...over,
  };
}

describe('classesStore — widget_tag is a normal class', () => {
  afterEach(() => {
    classesStore.classes = [];
  });

  it('a direct by-id lookup resolves widget_tag', () => {
    classesStore.classes = [
      cls({ id: 80, name: 'widget_tag', validated_count: 999_999 }),
    ];
    expect(classesStore.byId(80)?.name).toBe('widget_tag');
  });
});

// W4 (docs/design/logic-moves-adoption-plan-2026-09-24.md §2 W4): refresh()
// must populate thresholds/reservedHotkeys from the same /classes response
// that populates classes — classHotkey.ts's reservedHotkeyLetters() and
// adequacy.ts's chip rendering both read these store fields directly.
describe('classesStore.refresh — thresholds + reservedHotkeys', () => {
  afterEach(() => {
    classesStore.classes = [];
    classesStore.thresholds = {
      block_below: 0,
      warn_below: 0,
      min_test_per_class: 0,
      aug_target_min: 0,
      aug_target_max: 0,
    };
    classesStore.reservedHotkeys = [];
    vi.unstubAllGlobals();
  });

  it('stores the served thresholds and reserved_hotkeys alongside classes', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        ok({
          classes: [{ class_id: 8, class_name: 'widget_a', validated_count: 8 }],
          thresholds: {
            block_below: 20,
            warn_below: 500,
            min_test_per_class: 5,
            aug_target_min: 500,
            aug_target_max: 3000,
          },
          reserved_hotkeys: ['b', 'd'],
        }),
      ),
    );

    await classesStore.refresh();

    expect(classesStore.thresholds.block_below).toBe(20);
    expect(classesStore.reservedHotkeys).toEqual(['b', 'd']);
    expect(classesStore.classes).toHaveLength(1);
  });
});
