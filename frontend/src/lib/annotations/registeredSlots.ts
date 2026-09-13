/**
 * The one deployment-configuration file for which annotation slots are
 * actually live in this app (P2.10, docs/genericization-plan-2026-09-13.md
 * §3.3/Phase 3 ship-gate).
 *
 * This file lives outside `profiles/` on purpose — it is the single
 * "register a new domain here" edit point the whole plan exists to
 * create. Everything else (SlotCard/SlotGallery/ProvenanceChip/
 * BboxCanvas/SlotBboxEditor, reviewTabs.ts's REVIEW_TABS,
 * clusters/+page.svelte's class-filter routing, +layout.svelte's
 * sidebar-click routing) reads from `registeredSlots`/`slotRegistry`
 * instead of hardcoding `'license_plate'`, so adding a second live slot
 * is exactly: import its profile from `./profiles/`, add it to the
 * array below. No other production file changes.
 *
 * `aircraft_tail_number` and `defect_code` (profiles/aircraftTailNumber.ts,
 * profiles/defectCode.ts) are deliberately NOT in this list — they exist
 * only to prove the capability model is generic
 * (profiles.falsification.test.ts), not to actually ship as live
 * legacy-labeler features. Enabling either for real would also require
 * backend support (wire fields, endpoints) this deployment doesn't have.
 */

import { resolveSlotRegistry } from './registry';
import { licensePlateSlot } from './profiles/licensePlate';
import type { SlotSpec } from './types';

/** Slots active in this deployment, in display order. */
export const registeredSlots: SlotSpec[] = [licensePlateSlot];

export const { registry: slotRegistry, warnings: slotRegistryWarnings } =
  resolveSlotRegistry({ builtins: registeredSlots });

/** Case-insensitive class-name -> slot lookup, the common call-site shape
 *  (`class === 'license_plate'` checks used to do this by hand). */
export function slotForClassName(
  className: string | null | undefined,
): SlotSpec | undefined {
  const target = (className ?? '').toLowerCase();
  return registeredSlots.find((s) => (s.bind.className ?? '').toLowerCase() === target);
}
