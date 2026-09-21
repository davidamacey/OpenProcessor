/**
 * Bridges the annotation-slot adapter (`readSlot.ts`) into `api.ts`'s
 * crop mapping. Pure — no `api.ts` import, so it can be unit tested in
 * isolation and reused by any raw-row mapper (`mapRawCrop`, `getPlates`).
 *
 * See docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §4.3.
 */

import { slotRegistry } from './registeredSlots';
import { readSlot } from './readSlot';
import { slotIsPresent } from './types';
import type { SlotKey, SlotData, SlotSpec, XYXY } from './types';

/**
 * Maps every registry slot that has any evidence on this raw row.
 *
 * Reads `slotRegistry` at CALL time (not destructured at module scope)
 * so a tier-2 deployment profile installed after this module's initial
 * evaluation (`registeredSlots.ts:37-48` — the root layout's `load()`
 * runs after route modules import, but before render) is honored by
 * every call, not just calls made before `installDeploymentSlots` ran.
 */
export function mapCropSlots(
  raw: Record<string, unknown>,
  parentXyxy: XYXY,
): Record<SlotKey, SlotData> {
  const out: Record<SlotKey, SlotData> = {};
  for (const spec of slotRegistry.all) {
    const d = readSlot(raw, spec, parentXyxy);
    if (slotIsPresent(d)) out[spec.key] = d;
  }
  return out;
}

/**
 * The one documented way to get `SlotData` off an ALREADY-MAPPED crop.
 *
 * Never call `readSlot()` on a `OpCrop` — its bboxes are `BBoxNorm`
 * ({cx,cy,w,h}), not the `XYXY` tuples `readSlot` expects, so calling it
 * on a mapped crop would silently resolve every field to null instead of
 * throwing. `mapRawCrop` is the only place with the raw payload in hand;
 * every other consumer reads the crop's already-computed `slots` map
 * through this accessor.
 */
export function slotOf(
  crop: { slots?: Record<SlotKey, SlotData> },
  spec: SlotSpec | null | undefined,
): SlotData | null {
  return spec ? (crop.slots?.[spec.key] ?? null) : null;
}
