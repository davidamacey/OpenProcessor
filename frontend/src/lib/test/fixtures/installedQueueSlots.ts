/**
 * Installs the served widget-tag region profile and returns the registered
 * queue slots, failing when there are none. A loop over `registeredSlots`
 * at module scope iterates an empty list in the unit environment (no served
 * profile), so every assertion in it silently never ran; this makes the
 * non-empty precondition explicit. Pair with `afterEach(resetDeploymentSlots)`.
 */
import { expect } from 'vitest';
import {
  installServedRegionProfile,
  registeredSlots,
} from '$lib/annotations/registeredSlots';
import { WIDGET_TAG_PROFILE } from './regionSlot';

export function installedQueueSlots() {
  installServedRegionProfile(WIDGET_TAG_PROFILE);
  const slots = registeredSlots.filter((s) => s.capabilities.queue);
  expect(slots.length, 'a served profile registers a queue slot').toBeGreaterThan(0);
  return slots;
}
