import { describe, it, expect } from 'vitest';
import { resolveSlotRegistry } from './registry';
import { widgetTagSlot } from '$lib/test/fixtures/regionSlot';
import type { SlotSpec } from './types';

describe('resolveSlotRegistry', () => {
  it('resolves the built-in slot with no deployment override', () => {
    const { registry, warnings } = resolveSlotRegistry({ builtins: [widgetTagSlot] });
    expect(warnings).toEqual([]);
    expect(registry.all).toHaveLength(1);
    expect(registry.byKey('widget_tag')).toBe(widgetTagSlot);
    expect(registry.queues).toHaveLength(1);
  });

  it('forClass matches by className case-insensitively', () => {
    const { registry } = resolveSlotRegistry({ builtins: [widgetTagSlot] });
    const classes = new Map([[7, 'Widget_Tag']]);
    expect(registry.forClass(7, classes)).toEqual([widgetTagSlot]);
    expect(registry.forClass(8, classes)).toEqual([]);
  });

  it('deployment override REPLACES a built-in slot by key rather than deep-merging', () => {
    const override: SlotSpec = {
      ...widgetTagSlot,
      label: { singular: 'label', plural: 'labels', title: 'Label' },
      capabilities: { text: widgetTagSlot.capabilities.text },
    };
    const { registry, warnings } = resolveSlotRegistry({
      builtins: [widgetTagSlot],
      deployment: [override],
    });
    expect(warnings).toEqual([]);
    const resolved = registry.byKey('widget_tag');
    expect(resolved?.label.singular).toBe('label');
    // subBox capability from the built-in must NOT survive a partial override.
    expect(resolved?.capabilities.subBox).toBeUndefined();
  });

  it('a malformed deployment slot is dropped with a warning, built-ins survive', () => {
    const { registry, warnings } = resolveSlotRegistry({
      builtins: [widgetTagSlot],
      deployment: [{ key: 'broken' }, { notAKey: true }, null],
    });
    expect(warnings.length).toBeGreaterThan(0);
    expect(registry.all).toHaveLength(1);
    expect(registry.byKey('widget_tag')).toBe(widgetTagSlot);
    expect(registry.byKey('broken')).toBeUndefined();
  });
});
