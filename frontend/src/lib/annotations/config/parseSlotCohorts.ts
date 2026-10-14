/** `capabilities.trainingCohorts` parser: cohorts and their queries. */

import type {
  CohortSpec,
  CohortQuery,
  CohortFilter,
  CohortOp,
  TrainingCohortsCapability,
} from '../cohorts';
import { LIMITS, COHORT_PLACEHOLDERS } from './allowLists';
import { validatePathTemplate } from './templatePath';
import {
  hasForbiddenKey,
  isIdentifier,
  isLabel,
  isPlainObject,
  isWireField,
} from './parseCommon';

const COHORT_OPS: ReadonlySet<string> = new Set(['exists', 'eq', 'lt', 'containsAll']);
/* ------------------------------------------------------------------ */
/* trainingCohorts                                                     */
/* ------------------------------------------------------------------ */

function parseCohortQuery(
  raw: unknown,
  key: string,
  cohortId: string,
  errors: string[],
): CohortQuery | null {
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": cohort "${cohortId}".query must be an object — skipped`);
    return null;
  }
  if (raw.kind === 'endpoint') {
    const pathResult = validatePathTemplate(raw.path, COHORT_PLACEHOLDERS);
    if (!pathResult.template) {
      errors.push(
        `slot "${key}": cohort "${cohortId}".query.path — ${pathResult.error} — skipped`,
      );
      return null;
    }
    if (!isPlainObject(raw.params) || hasForbiddenKey(raw.params)) {
      errors.push(
        `slot "${key}": cohort "${cohortId}".query.params must be an object — skipped`,
      );
      return null;
    }
    const entries = Object.entries(raw.params);
    if (entries.length > LIMITS.cohortParams) {
      errors.push(
        `slot "${key}": cohort "${cohortId}".query.params exceeds ${LIMITS.cohortParams} entries — skipped`,
      );
      return null;
    }
    const params: Record<string, string | number | boolean> = {};
    for (const [pk, pv] of entries) {
      if (!isIdentifier(pk)) {
        errors.push(
          `slot "${key}": cohort "${cohortId}".query.params key "${pk}" is invalid — skipped`,
        );
        return null;
      }
      if (typeof pv === 'number' || typeof pv === 'boolean') {
        params[pk] = pv;
        continue;
      }
      if (typeof pv === 'string') {
        let sawBadPlaceholder = false;
        const stripped = pv.replace(/\{([^{}]*)\}/g, (_m, name: string) => {
          if (!COHORT_PLACEHOLDERS.includes(name)) sawBadPlaceholder = true;
          return '_';
        });
        if (sawBadPlaceholder || !/^[A-Za-z0-9._-]*$/.test(stripped)) {
          errors.push(
            `slot "${key}": cohort "${cohortId}".query.params.${pk} value is invalid — skipped`,
          );
          return null;
        }
        params[pk] = pv;
        continue;
      }
      errors.push(
        `slot "${key}": cohort "${cohortId}".query.params.${pk} must be a string, number, or boolean — skipped`,
      );
      return null;
    }
    return { kind: 'endpoint', path: raw.path as string, params };
  }
  if (raw.kind === 'predicate') {
    if (!Array.isArray(raw.filters)) {
      errors.push(
        `slot "${key}": cohort "${cohortId}".query.filters must be an array — skipped`,
      );
      return null;
    }
    const filters: CohortFilter[] = [];
    for (const f of raw.filters) {
      if (!isPlainObject(f) || hasForbiddenKey(f)) {
        errors.push(
          `slot "${key}": cohort "${cohortId}".query.filters entry must be an object — skipped`,
        );
        return null;
      }
      if (!isWireField(f.field)) {
        errors.push(
          `slot "${key}": cohort "${cohortId}".query.filters[].field is invalid — skipped`,
        );
        return null;
      }
      if (typeof f.op !== 'string' || !COHORT_OPS.has(f.op)) {
        errors.push(
          `slot "${key}": cohort "${cohortId}".query.filters[].op is invalid — skipped`,
        );
        return null;
      }
      const filter: CohortFilter = { field: f.field, op: f.op as CohortOp };
      if ('value' in f) filter.value = f.value as CohortFilter['value'];
      filters.push(filter);
    }
    const out: { kind: 'predicate'; filters: CohortFilter[]; excludeTestHoldout?: true } =
      {
        kind: 'predicate',
        filters,
      };
    if (raw.excludeTestHoldout !== undefined) {
      if (raw.excludeTestHoldout !== true) {
        errors.push(
          `slot "${key}": cohort "${cohortId}".query.excludeTestHoldout must be literally true — skipped`,
        );
        return null;
      }
      out.excludeTestHoldout = true;
    }
    return out;
  }
  errors.push(
    `slot "${key}": cohort "${cohortId}".query.kind must be "endpoint" or "predicate" — skipped`,
  );
  return null;
}

function parseCohort(
  raw: unknown,
  key: string,
  seenIds: Set<string>,
  errors: string[],
): CohortSpec | null {
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": a cohort entry must be an object — skipped`);
    return null;
  }
  const id = raw.id;
  if (!isIdentifier(id)) {
    errors.push(`slot "${key}": a cohort's id is invalid — skipped`);
    return null;
  }
  if (seenIds.has(id)) {
    errors.push(
      `slot "${key}": cohort id "${id}" is duplicated within this slot — skipped`,
    );
    return null;
  }
  if (!isLabel(raw.label)) {
    errors.push(`slot "${key}": cohort "${id}".label is invalid — skipped`);
    return null;
  }
  if (!isLabel(raw.description)) {
    errors.push(`slot "${key}": cohort "${id}".description is invalid — skipped`);
    return null;
  }
  const query = parseCohortQuery(raw.query, key, id, errors);
  if (query === null) return null;
  if (raw.rowKind !== 'slot' && raw.rowKind !== 'crop') {
    errors.push(
      `slot "${key}": cohort "${id}".rowKind must be "slot" or "crop" — skipped`,
    );
    return null;
  }
  const out: CohortSpec = {
    id,
    label: raw.label as string,
    description: raw.description as string,
    query,
    rowKind: raw.rowKind,
  };
  if (raw.reviewTarget !== undefined) {
    if (raw.reviewTarget !== 'slotQueue' && raw.reviewTarget !== 'all') {
      errors.push(
        `slot "${key}": cohort "${id}".reviewTarget must be "slotQueue" or "all" — skipped`,
      );
      return null;
    }
    out.reviewTarget = raw.reviewTarget;
  }
  seenIds.add(id);
  return out;
}

export function parseTrainingCohorts(
  raw: unknown,
  key: string,
  errors: string[],
): TrainingCohortsCapability | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(
      `slot "${key}": capabilities.trainingCohorts must be an object — skipped`,
    );
    return null;
  }
  const seenIds = new Set<string>();
  const cohorts: CohortSpec[] = [];
  if (raw.cohorts !== undefined) {
    if (!Array.isArray(raw.cohorts) || raw.cohorts.length > LIMITS.cohorts) {
      errors.push(
        `slot "${key}": capabilities.trainingCohorts.cohorts must be an array ≤ ${LIMITS.cohorts} — skipped`,
      );
      return null;
    }
    for (const c of raw.cohorts) {
      const parsed = parseCohort(c, key, seenIds, errors);
      if (parsed === null) return null;
      cohorts.push(parsed);
    }
  }
  let suppressDerived: string[] | undefined;
  if (raw.suppressDerived !== undefined) {
    if (
      !Array.isArray(raw.suppressDerived) ||
      !raw.suppressDerived.every((s: unknown) => isIdentifier(s))
    ) {
      errors.push(
        `slot "${key}": capabilities.trainingCohorts.suppressDerived must be an array of identifiers — skipped`,
      );
      return null;
    }
    suppressDerived = raw.suppressDerived as string[];
  }
  const out: TrainingCohortsCapability = { cohorts };
  if (suppressDerived !== undefined) out.suppressDerived = suppressDerived;
  return out;
}
