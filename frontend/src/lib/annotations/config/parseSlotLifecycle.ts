/** `capabilities.lifecycle` parser: the state list and its named states. */

import type { LifecycleCapability, SlotState } from '../types';
import { LIMITS, IDENTIFIER_RE } from './allowLists';
import { hasForbiddenKey, isLabel, isPlainObject, isWireField } from './parseCommon';

const STATE_ROLES: ReadonlySet<string> = new Set([
  'proposed',
  'confirmed',
  'rejected',
  'falsePositive',
  'absent',
  'pending',
]);
/* ------------------------------------------------------------------ */
/* lifecycle                                                           */
/* ------------------------------------------------------------------ */

const isIdentifierString = (v: unknown): v is string =>
  typeof v === 'string' && IDENTIFIER_RE.test(v) && v.length <= LIMITS.identifierChars;

const isValidAliases = (v: unknown): v is string[] =>
  Array.isArray(v) && v.length <= LIMITS.stateAliases && v.every(isIdentifierString);

/** Optional state fields: wire name, validity check, message suffix when invalid. */
const OPTIONAL_STATE_FIELDS = [
  {
    name: 'role',
    ok: (v: unknown) => typeof v === 'string' && STATE_ROLES.has(v),
    msg: 'is invalid',
  },
  { name: 'dim', ok: (v: unknown) => typeof v === 'boolean', msg: 'must be boolean' },
  { name: 'badge', ok: isLabel, msg: 'is invalid' },
  {
    name: 'aliases',
    ok: isValidAliases,
    msg: 'must be a string array of valid identifiers',
  },
] as const;

/** A state entry's first problem (message suffix after `states`), or null. */
function stateProblem(s: Record<string, unknown>, seen: Set<string>): string | null {
  if (!isIdentifierString(s.value)) return '[].value is invalid';
  if (seen.has(s.value)) return ` has duplicate value "${s.value}"`;
  seen.add(s.value);
  if (!isLabel(s.label)) return '[].label is invalid';
  if (typeof s.humanWritable !== 'boolean') return '[].humanWritable must be boolean';
  for (const f of OPTIONAL_STATE_FIELDS) {
    if (s[f.name] !== undefined && !f.ok(s[f.name])) return `[].${f.name} ${f.msg}`;
  }
  return null;
}

function parseStates(raw: unknown, key: string, errors: string[]): SlotState[] | null {
  const fail = (msg: string): null => {
    errors.push(`slot "${key}": capabilities.lifecycle.states${msg} — skipped`);
    return null;
  };
  if (!Array.isArray(raw) || raw.length === 0 || raw.length > LIMITS.lifecycleStates) {
    errors.push(
      `slot "${key}": capabilities.lifecycle.states must be a non-empty array ≤ ${LIMITS.lifecycleStates} — skipped`,
    );
    return null;
  }
  const seen = new Set<string>();
  const out: SlotState[] = [];
  for (const s of raw) {
    if (!isPlainObject(s) || hasForbiddenKey(s)) return fail(' entry must be an object');
    const problem = stateProblem(s, seen);
    if (problem) return fail(problem);
    const item = {
      value: s.value as string,
      label: s.label as string,
      humanWritable: s.humanWritable as boolean,
    } as SlotState;
    const loose = item as unknown as Record<string, unknown>;
    for (const f of OPTIONAL_STATE_FIELDS) {
      if (s[f.name] !== undefined) loose[f.name] = s[f.name];
    }
    out.push(item);
  }
  return out;
}

const LIFECYCLE_OPTIONAL_FIELDS = [
  'verifiedField',
  'rejectionReasonField',
  'labelSourceField',
] as const;

export function parseLifecycle(
  raw: unknown,
  key: string,
  errors: string[],
): LifecycleCapability | null | undefined {
  if (raw === undefined) return undefined;
  const fail = (msg: string): null => {
    errors.push(`slot "${key}": capabilities.lifecycle${msg} — skipped`);
    return null;
  };
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) return fail(' must be an object');
  if (!isWireField(raw.statusField)) return fail('.statusField is required');
  for (const f of LIFECYCLE_OPTIONAL_FIELDS) {
    if (raw[f] !== undefined && !isWireField(raw[f])) return fail(`.${f} is invalid`);
  }
  const states = parseStates(raw.states, key, errors);
  if (states === null) return null;
  const values = new Set(states.map((s) => s.value));
  const namesState = (v: unknown): boolean => typeof v === 'string' && values.has(v);
  if (!namesState(raw.confirmState)) {
    return fail('.confirmState must name a states[].value');
  }
  if (!namesState(raw.rejectState))
    return fail('.rejectState must name a states[].value');
  if (raw.falsePositiveState !== undefined && !namesState(raw.falsePositiveState)) {
    return fail('.falsePositiveState must name a states[].value');
  }
  const out: LifecycleCapability = {
    statusField: raw.statusField,
    states,
    confirmState: raw.confirmState as string,
    rejectState: raw.rejectState as string,
  };
  for (const f of [...LIFECYCLE_OPTIONAL_FIELDS, 'falsePositiveState'] as const) {
    if (raw[f] !== undefined) out[f] = raw[f] as string;
  }
  return out;
}
