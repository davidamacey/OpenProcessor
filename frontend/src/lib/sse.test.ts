/**
 * C7 (docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §7.1):
 * `subscribeCurationEvents` used to hardcode a fixed KNOWN_EVENT_TYPES list
 * (`crop.created` / `crop.classified` / `crop.region_verified`).
 * `EventSource.addEventListener` requires an exact type name, so any
 * event type NOT in that list is silently never dispatched — a second
 * queue-capable slot's own verify event (e.g. `crop.aircraft_tail_
 * number_verified`) would refresh nothing, with no error anywhere.
 *
 * `slotVerifiedEventTypes()` fixes this by deriving the list from
 * `slotRegistry.queues` at call time, plus the generic
 * `crop.region_verified` OpenProcessor emits for every region verify
 * regardless of slot key.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { slotVerifiedEventTypes } from './sse';
import {
  installDeploymentSlots,
  resetDeploymentSlots,
} from './annotations/registeredSlots';
import { aircraftTailNumberSlot } from './annotations/profiles/aircraftTailNumber';

afterEach(() => {
  resetDeploymentSlots();
});

describe('slotVerifiedEventTypes', () => {
  it('includes the generic crop.region_verified by default', () => {
    expect(slotVerifiedEventTypes()).toContain('crop.region_verified');
  });

  it('does NOT include a second slot event type before it is registered (the pre-fix trap)', () => {
    expect(slotVerifiedEventTypes()).not.toContain('crop.aircraft_tail_number_verified');
  });

  it('includes a second registered slot event type once installed, with zero further code change', () => {
    installDeploymentSlots([aircraftTailNumberSlot]);
    const types = slotVerifiedEventTypes();
    expect(types).toContain('crop.aircraft_tail_number_verified');
    expect(types).toContain('crop.region_verified');
  });

  it('reads slotRegistry at CALL time, not module-scope destructure', () => {
    const before = slotVerifiedEventTypes();
    expect(before).not.toContain('crop.aircraft_tail_number_verified');
    installDeploymentSlots([aircraftTailNumberSlot]);
    const after = slotVerifiedEventTypes();
    expect(after).toContain('crop.aircraft_tail_number_verified');
  });
});
