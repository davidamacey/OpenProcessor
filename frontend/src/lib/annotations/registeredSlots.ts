/**
 * The annotation slots live in this deployment (P2.10,
 * docs/genericization-plan-2026-09-13.md §3.3), composed from three
 * sources in order, merged per slot key with REPLACE semantics
 * (`resolveSlotRegistry`):
 *
 * 1. `builtinSlots` — slots compiled into the build (tier 1). Empty: the
 *    app ships with no domain built in.
 * 2. The served region slot — synthesized from the backend's
 *    `/health.region_profile` (`./servedRegionSlot.ts`,
 *    docs/design/domain-neutral-audit-2026-09-24.md §5). Absent when the
 *    backend has no region profile, which turns every region feature off.
 * 3. Deployment slots — `annotation-profiles.json` (tier 2,
 *    `./deploymentProfiles.ts`). A tier-2 entry whose key equals the
 *    served profile's name customizes the region slot (labels, keymap,
 *    cohorts); see `examples/annotation-profiles/`.
 *
 * The backend serves at most one region profile and 409s every region
 * route without one, so a tier-2 slot that calls region routes is dropped
 * (with a warning) when there is no served profile, or when its key is
 * not the served profile's name. A tier-2 slot that never touches a
 * region route is kept either way.
 *
 * Everything else (SlotCard/SlotGallery/SlotBboxEditor,
 * reviewTabs.ts's REVIEW_TABS, clusters/+page.svelte's class-filter
 * routing, +layout.svelte's sidebar routing) reads from
 * `registeredSlots`/`slotRegistry` rather than naming a slot.
 */

import { resolveSlotRegistry, type SlotRegistry } from './registry';
import { regionSlotFromServedProfile } from './servedRegionSlot';
import { normalizeClassName } from '$lib/classNameKey';
import type { SlotSpec } from './types';
import type { ServedRegionProfile } from '$lib/types';

/** Slots compiled into this build — tier 1. Never mutated. */
export const builtinSlots: SlotSpec[] = [];

/**
 * Slots active in this deployment, in display order.
 *
 * `let`, not `const`, and deliberately so: the served profile and tier-2
 * deployment profiles are fetched inside the root layout's `load()`,
 * which SvelteKit runs AFTER importing the route modules — i.e. after
 * this module body has already executed — but BEFORE rendering any
 * component. ESM live bindings mean every importer sees the merged list
 * at use time with no call-site change. See
 * docs/design/tier2-annotation-profile-config-plan-2026-09-20.md §1.5.
 */
export let registeredSlots: SlotSpec[] = builtinSlots;

const initial = resolveSlotRegistry({ builtins: builtinSlots });
export let slotRegistry: SlotRegistry = initial.registry;
export let slotRegistryWarnings: string[] = initial.warnings;
/** The subset of `slotRegistryWarnings` the region-profile drop rule
 *  produced (logged by the root layout once both sources resolve). */
export let regionRuleWarnings: string[] = [];

/** `undefined` until the root layout resolves `/health`: unknown, so no
 *  drop rule applies (unit tests that never load a profile). `null`
 *  once resolved with no profile configured. */
let servedProfile: ServedRegionProfile | null | undefined = undefined;
let deploymentSlots: SlotSpec[] = [];
let deploymentWarnings: string[] = [];

const listeners: Array<() => void> = [];

/** Called at module scope by anything that derives a module-level value
 *  from `registeredSlots` — today exactly one caller, `reviewTabs.ts`'s
 *  `REVIEW_TABS`. Registering here rather than having the installer
 *  import `reviewTabs.ts` keeps the dependency one-directional
 *  (`reviewTabs` → `registeredSlots`, never the reverse). */
export function onRegisteredSlotsChanged(cb: () => void): void {
  listeners.push(cb);
}

const REGION_ROUTE = /^\/regions(\/|\?|$)|^\/crops\/[^/]+\/region/;

/** Every backend path a slot declares, resolved with a placeholder id. */
function declaredPaths(slot: SlotSpec): string[] {
  const id = 'x';
  const e = slot.endpoints;
  const paths: Array<string | undefined> = [
    slot.capabilities.queue?.browsePath,
    slot.capabilities.subBox?.thumbnail?.path(id, 'b', 1),
    e.patchMeta?.(id),
    e.batchStatus?.(),
  ];
  for (const c of slot.capabilities.trainingCohorts?.cohorts ?? []) {
    if (c.query.kind === 'endpoint') paths.push(c.query.path);
  }
  return paths.filter((p): p is string => typeof p === 'string');
}

/** True when any path the slot declares is a region route — one the
 *  backend answers with 409 when it has no region profile. */
export function callsRegionRoutes(slot: SlotSpec): boolean {
  return declaredPaths(slot).some((p) => REGION_ROUTE.test(p));
}

/** A tier-2 slot that replaces the served region slot still gets the
 *  served write limit: a deployment file never declares it. */
function withServedLimits(slot: SlotSpec, profile: ServedRegionProfile): SlotSpec {
  const subBox = slot.capabilities.subBox;
  if (!subBox?.listField) return slot;
  return {
    ...slot,
    capabilities: {
      ...slot.capabilities,
      subBox: { ...subBox, maxBoxesPerWrite: profile.limits.max_boxes_per_write },
    },
  };
}

/**
 * The drop rule above, as a pure function: which deployment slots the
 * served region profile allows, and a warning for each one dropped.
 */
export function applyRegionProfileRule(
  slots: SlotSpec[],
  profile: ServedRegionProfile | null,
): { kept: SlotSpec[]; warnings: string[] } {
  const kept: SlotSpec[] = [];
  const warnings: string[] = [];
  for (const s of slots) {
    if (!callsRegionRoutes(s)) {
      kept.push(s);
    } else if (profile == null) {
      warnings.push(
        `slot "${s.key}" uses region routes, but the backend has no region profile configured — dropped`,
      );
    } else if (s.key !== profile.name) {
      warnings.push(
        `slot "${s.key}" uses region routes, but the backend's region profile is "${profile.name}" — dropped (use key "${profile.name}" to customize the region slot)`,
      );
    } else {
      kept.push(withServedLimits(s, profile));
    }
  }
  return { kept, warnings };
}

function recompute(): void {
  const served = servedProfile ? [regionSlotFromServedProfile(servedProfile)] : [];
  const rule =
    servedProfile === undefined
      ? { kept: deploymentSlots, warnings: [] }
      : applyRegionProfileRule(deploymentSlots, servedProfile);
  const { registry, warnings: mergeWarnings } = resolveSlotRegistry({
    builtins: [...builtinSlots, ...served],
    deployment: rule.kept,
  });
  slotRegistry = registry;
  registeredSlots = registry.all;
  regionRuleWarnings = rule.warnings;
  slotRegistryWarnings = [...deploymentWarnings, ...rule.warnings, ...mergeWarnings];
  for (const cb of listeners) cb();
}

/**
 * Installs the backend's served region profile (`null` = none
 * configured). Called once from the root layout's `load()`.
 */
export function installServedRegionProfile(profile: ServedRegionProfile | null): void {
  servedProfile = profile;
  recompute();
}

/**
 * Merges deployment-supplied slots over the built-in and served slots
 * and notifies derived-value listeners. Merge semantics are
 * `resolveSlotRegistry`'s, unchanged: per-key REPLACE, never deep-merge,
 * malformed entries dropped with a warning.
 */
export function installDeploymentSlots(
  deployment: SlotSpec[],
  warnings: string[] = [],
): void {
  deploymentSlots = deployment;
  deploymentWarnings = warnings;
  recompute();
}

/** Test-only: restores tier-1-only state (no served profile, no
 *  deployment slots). Reassigns straight back to the original `initial`
 *  resolution (and `builtinSlots` by reference) so a never-installed
 *  registry stays identical by reference (see `deploymentProfiles.test.ts`'s
 *  case 12). */
export function resetDeploymentSlots(): void {
  servedProfile = undefined;
  deploymentSlots = [];
  deploymentWarnings = [];
  slotRegistry = initial.registry;
  registeredSlots = builtinSlots;
  slotRegistryWarnings = initial.warnings;
  regionRuleWarnings = [];
  for (const cb of listeners) cb();
}

/** Case-insensitive class-name -> slot lookup, the common call-site shape
 *  (hand-written class-name checks used to do this). */
export function slotForClassName(
  className: string | null | undefined,
): SlotSpec | undefined {
  const target = normalizeClassName(className ?? '');
  if (!target) return undefined;
  return registeredSlots.find(
    (s) => normalizeClassName(s.bind.className ?? '') === target,
  );
}
