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
 * instead of hardcoding a slot key, so adding a second live slot
 * is exactly: import its profile from `./profiles/`, add it to the
 * array below. No other production file changes for a build-time slot.
 *
 * A DEPLOYMENT can also register a slot without a checkout at all — see
 * `./deploymentProfiles.ts` (tier 2, `static/annotation-profiles.json`).
 *
 * `aircraft_tail_number` and `defect_code` (profiles/aircraftTailNumber.ts,
 * profiles/defectCode.ts) are deliberately NOT in this list — they exist
 * only to prove the capability model is generic
 * (profiles.falsification.test.ts), not to actually ship as live
 * Cropwright features. Enabling either for real would also require
 * backend support (wire fields, endpoints) this deployment doesn't have.
 */

import { resolveSlotRegistry, type SlotRegistry } from './registry';
import { licensePlateSlot } from './profiles/licensePlate';
import type { SlotSpec } from './types';

/** Slots compiled into this build — tier 1. Never mutated. */
export const builtinSlots: SlotSpec[] = [licensePlateSlot];

/**
 * Slots active in this deployment, in display order.
 *
 * `let`, not `const`, and deliberately so: tier-2 deployment profiles
 * (`./deploymentProfiles.ts`) are fetched inside the root layout's
 * `load()`, which SvelteKit runs AFTER importing the route modules —
 * i.e. after this module body has already executed — but BEFORE
 * rendering any component. ESM live bindings mean every importer sees
 * the merged list at use time with no call-site change. See
 * docs/design/tier2-annotation-profile-config-plan-2026-09-20.md §1.5
 * for the three options considered and why this one.
 *
 * Re-assigned exactly once, by `installDeploymentSlots`, and only when
 * a deployment profile actually produced slots or warnings.
 */
export let registeredSlots: SlotSpec[] = builtinSlots;

const initial = resolveSlotRegistry({ builtins: builtinSlots });
export let slotRegistry: SlotRegistry = initial.registry;
export let slotRegistryWarnings: string[] = initial.warnings;

const listeners: Array<() => void> = [];

/** Called at module scope by anything that derives a module-level value
 *  from `registeredSlots` — today exactly one caller, `reviewTabs.ts`'s
 *  `REVIEW_TABS`. Registering here rather than having the installer
 *  import `reviewTabs.ts` keeps the dependency one-directional
 *  (`reviewTabs` → `registeredSlots`, never the reverse). */
export function onRegisteredSlotsChanged(cb: () => void): void {
  listeners.push(cb);
}

/**
 * Merges deployment-supplied slots over the built-ins and notifies
 * derived-value listeners. Merge semantics are `resolveSlotRegistry`'s,
 * unchanged: per-key REPLACE, never deep-merge, malformed entries
 * dropped with a warning.
 */
export function installDeploymentSlots(
  deployment: SlotSpec[],
  warnings: string[] = [],
): void {
  const { registry, warnings: mergeWarnings } = resolveSlotRegistry({
    builtins: builtinSlots,
    deployment,
  });
  slotRegistry = registry;
  registeredSlots = registry.all;
  slotRegistryWarnings = [...warnings, ...mergeWarnings];
  for (const cb of listeners) cb();
}

/** Test-only: restores tier-1-only state. Reassigns straight back to the
 *  original `initial` resolution (and `builtinSlots` by reference) rather
 *  than routing through `installDeploymentSlots([], [])`, which would
 *  recompute an equivalent-but-distinct array from `resolveSlotRegistry`
 *  — breaking the "identical by reference" property a never-installed
 *  deployment profile is supposed to preserve (see
 *  `deploymentProfiles.test.ts`'s case 12). */
export function resetDeploymentSlots(): void {
  slotRegistry = initial.registry;
  registeredSlots = builtinSlots;
  slotRegistryWarnings = initial.warnings;
  for (const cb of listeners) cb();
}

/** Case-insensitive class-name -> slot lookup, the common call-site shape
 *  (hand-written class-name checks used to do this). */
export function slotForClassName(
  className: string | null | undefined,
): SlotSpec | undefined {
  const target = (className ?? '').toLowerCase();
  return registeredSlots.find((s) => (s.bind.className ?? '').toLowerCase() === target);
}
