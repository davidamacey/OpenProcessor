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

type ParamValue = string | number | boolean;

/** True when a string param value only uses known `{placeholder}`s and safe chars. */
function isSafeParamString(pv: string): boolean {
  let sawBadPlaceholder = false;
  const stripped = pv.replace(/\{([^{}]*)\}/g, (_m, name: string) => {
    if (!COHORT_PLACEHOLDERS.includes(name)) sawBadPlaceholder = true;
    return '_';
  });
  return !sawBadPlaceholder && /^[A-Za-z0-9._-]*$/.test(stripped);
}

/** Why a query param is rejected (message suffix), or null when it is fine. */
function paramProblem(pk: string, pv: unknown): string | null {
  if (!isIdentifier(pk)) return `params key "${pk}" is invalid`;
  if (typeof pv === 'number' || typeof pv === 'boolean') return null;
  if (typeof pv === 'string') {
    return isSafeParamString(pv) ? null : `params.${pk} value is invalid`;
  }
  return `params.${pk} must be a string, number, or boolean`;
}

function parseEndpointQuery(
  raw: Record<string, unknown>,
  fail: (msg: string) => null,
): CohortQuery | null {
  const pathResult = validatePathTemplate(raw.path, COHORT_PLACEHOLDERS);
  if (!pathResult.template) return fail(`path — ${pathResult.error}`);
  if (!isPlainObject(raw.params) || hasForbiddenKey(raw.params)) {
    return fail('params must be an object');
  }
  const entries = Object.entries(raw.params);
  if (entries.length > LIMITS.cohortParams) {
    return fail(`params exceeds ${LIMITS.cohortParams} entries`);
  }
  const params: Record<string, ParamValue> = {};
  for (const [pk, pv] of entries) {
    const problem = paramProblem(pk, pv);
    if (problem) return fail(problem);
    params[pk] = pv as ParamValue;
  }
  return { kind: 'endpoint', path: raw.path as string, params };
}

function filterProblem(f: unknown): string | null {
  if (!isPlainObject(f) || hasForbiddenKey(f)) return 'filters entry must be an object';
  if (!isWireField(f.field)) return 'filters[].field is invalid';
  if (typeof f.op !== 'string' || !COHORT_OPS.has(f.op)) return 'filters[].op is invalid';
  return null;
}

function parsePredicateQuery(
  raw: Record<string, unknown>,
  fail: (msg: string) => null,
): CohortQuery | null {
  if (!Array.isArray(raw.filters)) return fail('filters must be an array');
  const filters: CohortFilter[] = [];
  for (const f of raw.filters) {
    const problem = filterProblem(f);
    if (problem) return fail(problem);
    const rec = f as Record<string, unknown>;
    const filter: CohortFilter = { field: rec.field as string, op: rec.op as CohortOp };
    if ('value' in rec) filter.value = rec.value as CohortFilter['value'];
    filters.push(filter);
  }
  const out: { kind: 'predicate'; filters: CohortFilter[]; excludeTestHoldout?: true } = {
    kind: 'predicate',
    filters,
  };
  if (raw.excludeTestHoldout !== undefined) {
    if (raw.excludeTestHoldout !== true) {
      return fail('excludeTestHoldout must be literally true');
    }
    out.excludeTestHoldout = true;
  }
  return out;
}

function parseCohortQuery(
  raw: unknown,
  key: string,
  cohortId: string,
  errors: string[],
): CohortQuery | null {
  const fail = (msg: string): null => {
    errors.push(`slot "${key}": cohort "${cohortId}".query.${msg} — skipped`);
    return null;
  };
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) return fail('must be an object');
  if (raw.kind === 'endpoint') return parseEndpointQuery(raw, fail);
  if (raw.kind === 'predicate') return parsePredicateQuery(raw, fail);
  return fail('kind must be "endpoint" or "predicate"');
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
