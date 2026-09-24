import { describe, it, expect } from 'vitest';
import { resolveSlotRegistry } from './registry';
import { licensePlateSlot } from './profiles/licensePlate';
import type { SlotSpec } from './types';

describe('resolveSlotRegistry', () => {
  it('resolves the built-in slot with no deployment override', () => {
    const { registry, warnings } = resolveSlotRegistry({ builtins: [licensePlateSlot] });
    expect(warnings).toEqual([]);
    expect(registry.all).toHaveLength(1);
    expect(registry.byKey('license_plate')).toBe(licensePlateSlot);
    expect(registry.queues).toHaveLength(1);
  });

  it('forClass matches by className case-insensitively', () => {
    const { registry } = resolveSlotRegistry({ builtins: [licensePlateSlot] });
    const classes = new Map([[7, 'License_Plate']]);
    expect(registry.forClass(7, classes)).toEqual([licensePlateSlot]);
    expect(registry.forClass(8, classes)).toEqual([]);
  });

  it('deployment override REPLACES a built-in slot by key rather than deep-merging', () => {
    const override: SlotSpec = {
      ...licensePlateSlot,
      label: { singular: 'tag', plural: 'tags', title: 'Tag' },
      capabilities: { text: licensePlateSlot.capabilities.text },
    };
    const { registry, warnings } = resolveSlotRegistry({
      builtins: [licensePlateSlot],
      deployment: [override],
    });
    expect(warnings).toEqual([]);
    const resolved = registry.byKey('license_plate');
    expect(resolved?.label.singular).toBe('tag');
    // subBox capability from the built-in must NOT survive a partial override.
    expect(resolved?.capabilities.subBox).toBeUndefined();
  });

  it('a malformed deployment slot is dropped with a warning, built-ins survive', () => {
    const { registry, warnings } = resolveSlotRegistry({
      builtins: [licensePlateSlot],
      deployment: [{ key: 'broken' }, { notAKey: true }, null],
    });
    expect(warnings.length).toBeGreaterThan(0);
    expect(registry.all).toHaveLength(1);
    expect(registry.byKey('license_plate')).toBe(licensePlateSlot);
    expect(registry.byKey('broken')).toBeUndefined();
  });
});
