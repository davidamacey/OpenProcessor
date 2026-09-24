/**
 * Resolves the deployed set of annotation slots.
 *
 * Merge order (docs/genericization-plan-2026-09-13.md §2.3): built-ins
 * ← deployment override (`static/annotation-profiles.json`, tier 2 —
 * fetched at boot by `./deploymentProfiles.ts` and merged in via the
 * root layout, see `docs/design/tier2-annotation-profile-config-plan-
 * 2026-09-20.md`) ← server (tier 3, `RegistryClass.annotation_slots`, still
 * not wired — deliberately deferred until tier 2 proves the schema in
 * production, per that plan's §9 sequencing). Merging is per-slot-key
 * REPLACE, not deep-merge, so a partial override can't silently
 * half-apply a broken profile.
 */

import type { SlotSpec, SlotKey } from './types';

export interface SlotRegistry {
  /** All resolved slots, deployment order. */
  all: SlotSpec[];
  /** Slots bound to a given class id. Today: at most 1. */
  forClass(classId: number, classesById: Map<number, string>): SlotSpec[];
  byKey(key: SlotKey): SlotSpec | undefined;
  /** Slots with a `queue` capability, in registry order. */
  queues: SlotSpec[];
}

function validateSlot(spec: unknown, warnings: string[]): spec is SlotSpec {
  if (typeof spec !== 'object' || spec === null) {
    warnings.push('slot spec is not an object — skipped');
    return false;
  }
  const s = spec as Record<string, unknown>;
  if (typeof s.key !== 'string' || s.key.length === 0) {
    warnings.push('slot spec missing a non-empty string `key` — skipped');
    return false;
  }
  if (typeof s.bind !== 'object' || s.bind === null) {
    warnings.push(`slot "${s.key}" missing \`bind\` — skipped`);
    return false;
  }
  const bind = s.bind as Record<string, unknown>;
  if (bind.className == null && bind.classId == null) {
    warnings.push(
      `slot "${s.key}" has neither bind.className nor bind.classId — skipped`,
    );
    return false;
  }
  if (typeof s.capabilities !== 'object' || s.capabilities === null) {
    warnings.push(`slot "${s.key}" missing \`capabilities\` — skipped`);
    return false;
  }
  return true;
}

export function resolveSlotRegistry(opts: {
  builtins: SlotSpec[];
  deployment?: unknown[];
}): { registry: SlotRegistry; warnings: string[] } {
  const warnings: string[] = [];
  const byKey = new Map<SlotKey, SlotSpec>();

  for (const spec of opts.builtins) {
    byKey.set(spec.key, spec);
  }

  for (const candidate of opts.deployment ?? []) {
    if (validateSlot(candidate, warnings)) {
      byKey.set(candidate.key, candidate);
    }
  }

  const all = [...byKey.values()];
  const queues = all.filter((s) => s.capabilities.queue);

  const registry: SlotRegistry = {
    all,
    byKey: (key) => byKey.get(key),
    queues,
    forClass(classId, classesById) {
      const className = classesById.get(classId)?.toLowerCase();
      return all.filter((s) => {
        if (s.bind.classId != null) return s.bind.classId === classId;
        if (s.bind.className != null) return s.bind.className.toLowerCase() === className;
        return false;
      });
    },
  };

  return { registry, warnings };
}
