/**
 * Registry composition around the served region profile
 * (docs/design/domain-neutral-audit-2026-09-24.md §5.2, §5.4): no
 * profile means no region slot, and a tier-2 slot that calls region
 * routes survives only under the served profile's own key.
 */
import { afterEach, describe, expect, it } from 'vitest';
import {
  applyRegionProfileRule,
  builtinSlots,
  callsRegionRoutes,
  installDeploymentSlots,
  installServedRegionProfile,
  registeredSlots,
  resetDeploymentSlots,
  slotForClassName,
  slotRegistry,
  slotRegistryWarnings,
} from './registeredSlots';
import type { SlotSpec } from './types';
import { aircraftTailNumberSlot } from '$lib/test/fixtures/aircraftTailNumberSlot';
import { defectCodeSlot } from '$lib/test/fixtures/defectCodeSlot';
import {
  WIDGET_TAG_CLASS,
  WIDGET_TAG_PROFILE,
  widgetTagSlot,
} from '$lib/test/fixtures/regionSlot';

afterEach(() => resetDeploymentSlots());

describe('the build ships no domain', () => {
  it('builtinSlots is empty', () => {
    expect(builtinSlots).toEqual([]);
  });
});

describe('no region profile configured', () => {
  it('registers no slot at all', () => {
    installServedRegionProfile(null);
    expect(registeredSlots).toEqual([]);
    expect(slotRegistry.queues).toEqual([]);
    expect(slotForClassName(WIDGET_TAG_CLASS)).toBeUndefined();
    expect(slotForClassName('anything')).toBeUndefined();
    expect(slotRegistryWarnings).toEqual([]);
  });

  it('drops a tier-2 slot that calls region routes, with a warning', () => {
    installServedRegionProfile(null);
    installDeploymentSlots([widgetTagSlot]);
    expect(registeredSlots).toEqual([]);
    expect(slotRegistryWarnings).toHaveLength(1);
    expect(slotRegistryWarnings[0]).toMatch(/no region profile/);
  });

  it('keeps a tier-2 slot that never touches a region route', () => {
    installServedRegionProfile(null);
    installDeploymentSlots([defectCodeSlot, aircraftTailNumberSlot]);
    expect(registeredSlots.map((s) => s.key)).toEqual([
      'defect_code',
      'aircraft_tail_number',
    ]);
    expect(slotRegistryWarnings).toEqual([]);
  });

  it('is order-independent: a deployment installed before the profile resolves is re-filtered', () => {
    installDeploymentSlots([widgetTagSlot]);
    expect(registeredSlots.map((s) => s.key)).toEqual(['widget_tag']);
    installServedRegionProfile(null);
    expect(registeredSlots).toEqual([]);
  });
});

describe('a region profile is configured', () => {
  it('registers the synthesized region slot, bound to the served class and labelled by display_name', () => {
    installServedRegionProfile(WIDGET_TAG_PROFILE);
    expect(registeredSlots).toHaveLength(1);
    const slot = slotForClassName(WIDGET_TAG_CLASS)!;
    expect(slot.key).toBe(WIDGET_TAG_PROFILE.name);
    expect(slot.capabilities.queue?.tabLabel).toBe(WIDGET_TAG_PROFILE.display_name);
  });

  it('a tier-2 slot keyed on the profile name replaces the served slot wholesale', () => {
    installServedRegionProfile(WIDGET_TAG_PROFILE);
    installDeploymentSlots([widgetTagSlot]);
    expect(registeredSlots).toEqual([widgetTagSlot]);
    expect(slotRegistryWarnings).toEqual([]);
  });

  it('drops a tier-2 region slot under any other key, with a warning naming the served profile', () => {
    installServedRegionProfile(WIDGET_TAG_PROFILE);
    const other: SlotSpec = { ...widgetTagSlot, key: 'another_region' };
    installDeploymentSlots([other]);
    expect(registeredSlots.map((s) => s.key)).toEqual([WIDGET_TAG_PROFILE.name]);
    expect(slotRegistryWarnings).toHaveLength(1);
    expect(slotRegistryWarnings[0]).toContain(`"${WIDGET_TAG_PROFILE.name}"`);
  });
});

describe('callsRegionRoutes / applyRegionProfileRule', () => {
  it('recognizes every region route shape a slot can declare', () => {
    const base = { ...defectCodeSlot, endpoints: {} };
    const withBrowse = (browsePath: string): SlotSpec => ({
      ...base,
      capabilities: {
        ...base.capabilities,
        queue: { ...base.capabilities.queue!, browsePath },
      },
    });
    expect(callsRegionRoutes(withBrowse('/regions'))).toBe(true);
    expect(callsRegionRoutes(withBrowse('/regions?x=1'))).toBe(true);
    expect(callsRegionRoutes(withBrowse('/defects'))).toBe(false);
    expect(callsRegionRoutes(withBrowse('/regionsx'))).toBe(false);
    expect(
      callsRegionRoutes({
        ...base,
        endpoints: { patchMeta: (id) => `/crops/${id}/region_meta` },
      }),
    ).toBe(true);
    expect(
      callsRegionRoutes({
        ...base,
        endpoints: { batchStatus: () => '/regions/batch_status' },
      }),
    ).toBe(true);
    expect(callsRegionRoutes(defectCodeSlot)).toBe(false);
    expect(callsRegionRoutes(widgetTagSlot)).toBe(true);
  });

  it('keeps everything that is not a region slot and nothing else when there is no profile', () => {
    const { kept, warnings } = applyRegionProfileRule(
      [widgetTagSlot, defectCodeSlot],
      null,
    );
    expect(kept).toEqual([defectCodeSlot]);
    expect(warnings).toHaveLength(1);
  });
});
