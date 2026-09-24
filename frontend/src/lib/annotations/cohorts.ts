/**
 * Training cohorts — a named, previewable slice of crops offered as the
 * input to a training run (docs/genericization-plan-2026-09-13.md
 * §9.2, addendum 2026-09-14). Deliberately NOT a general query
 * language: the union below has exactly two arms, and the `predicate`
 * arm's operator set is closed at the four operators every existing
 * region training-candidate mode actually uses (§9.1's mode table).
 *
 * Tier 1 (`endpoint`) requires NO backend change — it is today's
 * `?mode=` call (or `GET {API_PREFIX}/crops`'s existing class-agnostic params)
 * expressed as data. Tier 2 (`predicate`) is the H5 contract offer
 * (§9.4) and is gated at runtime, exactly like the `diverse`/
 * `viz_projection` overlays in `strategies.ts`: invisible until the
 * backend says it exists.
 */

import type { SlotKey, SlotSpec, WireField } from './types';
import type { SlotRegistry } from './registry';

/** Placeholder set is ALLOW-LISTED, mirroring §2.9's template-string
 *  compilation. A deployment profile loaded from JSON can interpolate
 *  these and nothing else — no arbitrary expressions, no field access. */
export type CohortTemplate = string; // e.g. '{classId}', '{slotKey}'

export interface CohortContext {
  classId: number;
  className: string;
  slotKey?: SlotKey;
}

/** Tier 1 — a call the backend already answers. */
export interface CohortEndpointQuery {
  kind: 'endpoint';
  /** Relative to API_PREFIX, joined by apiFetch — never absolute (§2.5). */
  path: CohortTemplate;
  params: Record<string, CohortTemplate | number | boolean>;
}

/** Tier 2 — a predicate the backend compiles. Requires H5. */
export type CohortOp = 'exists' | 'eq' | 'lt' | 'containsAll';

export interface CohortFilter {
  field: WireField;
  op: CohortOp;
  /** Absent for `exists`; string[] for `containsAll`. */
  value?: string | number | boolean | string[];
}

export interface CohortPredicateQuery {
  kind: 'predicate';
  filters: CohortFilter[];
  /** Always applied by the server regardless; listed for documentation
   *  parity with `_training_candidate_query`'s shared `must_not`. */
  excludeTestHoldout?: true;
}

export type CohortQuery = CohortEndpointQuery | CohortPredicateQuery;

export interface CohortSpec {
  /** Stable id. For a region training-candidate cohort this MUST equal
   *  the backend `mode` string so the call is byte-identical (§9.2.3). */
  id: string;
  label: string;
  description: string;
  query: CohortQuery;
  /**
   * Row shape the preview grid should expect. `'slot'` ⇒ rows carry the
   * slot's wire fields and render via SlotCard; `'crop'` ⇒ plain crops,
   * render via CropCard. Fixes the unconditional `<SlotCard>` at the
   * old train/+page.svelte:891.
   */
  rowKind: 'slot' | 'crop';
  /** Where a preview card click should land. Slot cohorts jump to the
   *  slot's queue tab; crop cohorts to `?tab=all`. */
  reviewTarget?: 'slotQueue' | 'all';
}

export interface TrainingCohortsCapability {
  /**
   * Explicit cohorts for this slot. These are REPLACEMENTS-by-id for
   * anything `derivedCohorts()` would derive — which is how a slot keeps
   * its own hand-tuned server-side modes instead of getting the weaker
   * generic versions.
   */
  cohorts: CohortSpec[];
  /**
   * Suppress specific derived ids without declaring a replacement.
   * Escape hatch for "this slot genuinely has no meaningful X cohort."
   */
  suppressDerived?: string[];
}

/* ------------------------------------------------------------------ */
/* Class-agnostic core cohorts                                         */
/* ------------------------------------------------------------------ */

/**
 * Not a slot capability — hangs off a class, so a class with no
 * registered slot at all still has something to train on. Rides
 * entirely on `GET {API_PREFIX}/crops` (already accepts `class_id`,
 * `label_validated`, `classifier_conf_lt`, and has a real response model) plus
 * the review surface's `model_disagreements` cohort, already
 * class-filterable. **Zero backend change.**
 */
export const CORE_COHORTS: CohortSpec[] = [
  {
    id: 'validated',
    label: 'Validated',
    description: 'Human-confirmed labels for this class — the default training pool',
    query: {
      kind: 'endpoint',
      path: '/crops',
      params: { class_id: '{classId}', label_validated: true },
    },
    rowKind: 'crop',
    reviewTarget: 'all',
  },
  {
    id: 'needs_labeling',
    label: 'Needs labeling',
    description: 'Assigned to this class but never human-validated',
    query: {
      kind: 'endpoint',
      path: '/crops',
      params: { class_id: '{classId}', label_validated: false },
    },
    rowKind: 'crop',
    reviewTarget: 'all',
  },
  {
    id: 'low_confidence',
    label: 'Low confidence',
    description: 'Primary-model confidence below 0.5 — high-loss rows',
    query: {
      kind: 'endpoint',
      path: '/crops',
      params: { class_id: '{classId}', classifier_conf_lt: 0.5 },
    },
    rowKind: 'crop',
    reviewTarget: 'all',
  },
  {
    id: 'model_disagreements',
    label: 'Model disagreements',
    description: 'Validated crops where the promoted model disagrees with the human',
    query: {
      kind: 'endpoint',
      path: '/review/model_disagreements',
      params: { class_id: '{classId}' },
    },
    rowKind: 'crop',
    reviewTarget: 'all',
  },
];

/* ------------------------------------------------------------------ */
/* Derived cohorts — capability -> cohort, for a slot with no          */
/* trainingCohorts entry of its own                                    */
/* ------------------------------------------------------------------ */

/**
 * Capability ⇒ cohort derivation. Each rule states the capability it
 * needs and the predicate it means, in the slot's OWN wire-field names —
 * which is why a generic blind-spots cohort is expressible at all.
 *
 * A slot's own hand-declared cohorts (`capabilities.trainingCohorts.
 * cohorts`) override every id here — "derived is the floor, declared is
 * the ceiling." All entries here are tier-2 `predicate` queries and are
 * therefore invisible until H5 lands (`predicateCohortsAvailable`),
 * so this adds nothing to today's UI and cannot regress it.
 */
export function derivedCohorts(spec: SlotSpec): CohortSpec[] {
  const out: CohortSpec[] = [];
  const { subBox, provenance, lifecycle } = spec.capabilities;

  // subBox ⇒ "parent subject is present, sub-box is not" — the generic
  // meaning of a blind spot for any sub-annotation.
  if (subBox) {
    out.push({
      id: 'blind_spots',
      label: `${spec.label.title} blind spots`,
      description: `Crops of this class with no ${spec.label.singular} box detected`,
      query: {
        kind: 'predicate',
        filters: [{ field: subBox.bboxField, op: 'exists', value: false }],
        excludeTestHoldout: true,
      },
      rowKind: 'crop',
      reviewTarget: 'all',
    });
  }

  // subBox + score ⇒ low-confidence detections.
  if (subBox?.scoreField) {
    out.push({
      id: 'low_conf',
      label: `Low-confidence ${spec.label.plural}`,
      description: `Detector score below 0.6`,
      query: {
        kind: 'predicate',
        filters: [{ field: subBox.scoreField, op: 'lt', value: 0.6 }],
        excludeTestHoldout: true,
      },
      rowKind: 'slot',
      reviewTarget: 'slotQueue',
    });
  }

  // provenance.chainField ⇒ two detectors both fired; the cascade log is
  // the only evidence of a disagreement.
  if (provenance?.chainField) {
    out.push({
      id: 'disagreement',
      label: 'Detector disagreements',
      description: 'More than one detector in the cascade produced a hit',
      query: {
        kind: 'predicate',
        filters: [{ field: provenance.chainField, op: 'exists' }],
        excludeTestHoldout: true,
      },
      rowKind: 'slot',
      reviewTarget: 'slotQueue',
    });
  }

  // lifecycle.falsePositiveState ⇒ a human-curated hard-negative pool.
  if (lifecycle?.falsePositiveState) {
    out.push({
      id: 'false_positives',
      label: 'False positives',
      description: 'Human-marked false positives — hard negatives',
      query: {
        kind: 'predicate',
        filters: [
          { field: lifecycle.statusField, op: 'eq', value: lifecycle.falsePositiveState },
        ],
        excludeTestHoldout: true,
      },
      rowKind: 'slot',
      reviewTarget: 'slotQueue',
    });
  }

  return out;
}

/* ------------------------------------------------------------------ */
/* Template compilation + resolution                                   */
/* ------------------------------------------------------------------ */

function substitute(
  value: CohortTemplate | number | boolean,
  ctx: CohortContext,
): CohortTemplate | number | boolean {
  if (typeof value !== 'string') return value;
  return value
    .replaceAll('{classId}', String(ctx.classId))
    .replaceAll('{slotKey}', ctx.slotKey ?? '');
}

/** Compiles every `{classId}`/`{slotKey}` placeholder in a cohort's
 *  query against a resolved class/slot context. Tier-2 `predicate`
 *  queries carry no templates in this design (their filters are
 *  spelled out in the slot's own wire-field names already) and pass
 *  through unchanged. */
export function compileCohortQuery(query: CohortQuery, ctx: CohortContext): CohortQuery {
  if (query.kind === 'predicate') return query;
  const params: Record<string, CohortTemplate | number | boolean> = {};
  for (const [key, value] of Object.entries(query.params)) {
    params[key] = substitute(value, ctx);
  }
  return { kind: 'endpoint', path: substitute(query.path, ctx) as string, params };
}

function compiled(spec: CohortSpec, ctx: CohortContext): CohortSpec {
  return { ...spec, query: compileCohortQuery(spec.query, ctx) };
}

/**
 * The three tier-1 endpoint shapes `/train`'s `runCohortQuery` knows how
 * to answer today (§9.1's mode table + §9.2.2's CORE_COHORTS), keyed by
 * a compiled query's final path segment rather than the whole path.
 *
 * Matching the last segment rather than the whole path (Wave 2 C12,
 * docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §10) means a
 * backend base-path rename, or a second slot's endpoint cohort under
 * another base path (e.g. a tier-2 profile's
 * `/widgets/training_candidates`), still dispatches instead of silently
 * returning `{total: 0, items: []}` — the dispatch cares only what kind
 * of endpoint a path names.
 */
export type CohortEndpointKind = 'training_candidates' | 'crops' | 'model_disagreements';

export function cohortEndpointKind(path: string): CohortEndpointKind | null {
  const segment = path.split('/').filter(Boolean).pop();
  if (segment === 'training_candidates') return 'training_candidates';
  if (segment === 'model_disagreements') return 'model_disagreements';
  if (segment === 'crops') return 'crops';
  return null;
}

/**
 * Resolution order: `CORE_COHORTS` (per class) ++ derived ++ declared,
 * later entries replacing earlier ones by `id`, minus `suppressDerived`.
 * Tier-2 entries are dropped entirely when the backend flag is off, so
 * a deployment with no H5 support renders exactly `CORE_COHORTS` plus
 * whatever each bound slot explicitly declares — forever.
 */
export function cohortsForClass(
  classId: number,
  className: string,
  registry: SlotRegistry,
  classesById: Map<number, string>,
  predicateCohortsAvailable: boolean,
): CohortSpec[] {
  const ctx: CohortContext = { classId, className };
  const byId = new Map<string, CohortSpec>();

  for (const core of CORE_COHORTS) byId.set(core.id, compiled(core, ctx));

  for (const slot of registry.forClass(classId, classesById)) {
    const slotCtx: CohortContext = { ...ctx, slotKey: slot.key };
    const training = slot.capabilities.trainingCohorts;
    const suppressed = new Set(training?.suppressDerived ?? []);

    if (predicateCohortsAvailable) {
      for (const derived of derivedCohorts(slot)) {
        if (suppressed.has(derived.id)) continue;
        byId.set(derived.id, compiled(derived, slotCtx));
      }
    }

    for (const declared of training?.cohorts ?? []) {
      byId.set(declared.id, compiled(declared, slotCtx));
    }
  }

  return [...byId.values()];
}
