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

function parseStates(raw: unknown, key: string, errors: string[]): SlotState[] | null {
  if (!Array.isArray(raw) || raw.length === 0 || raw.length > LIMITS.lifecycleStates) {
    errors.push(
      `slot "${key}": capabilities.lifecycle.states must be a non-empty array ≤ ${LIMITS.lifecycleStates} — skipped`,
    );
    return null;
  }
  const seen = new Set<string>();
  const out: SlotState[] = [];
  for (const s of raw) {
    if (!isPlainObject(s) || hasForbiddenKey(s)) {
      errors.push(
        `slot "${key}": capabilities.lifecycle.states entry must be an object — skipped`,
      );
      return null;
    }
    const { value, label, humanWritable } = s;
    if (
      typeof value !== 'string' ||
      !IDENTIFIER_RE.test(value) ||
      value.length > LIMITS.identifierChars
    ) {
      errors.push(
        `slot "${key}": capabilities.lifecycle.states[].value is invalid — skipped`,
      );
      return null;
    }
    if (seen.has(value)) {
      errors.push(
        `slot "${key}": capabilities.lifecycle.states has duplicate value "${value}" — skipped`,
      );
      return null;
    }
    seen.add(value);
    if (!isLabel(label)) {
      errors.push(
        `slot "${key}": capabilities.lifecycle.states[].label is invalid — skipped`,
      );
      return null;
    }
    if (typeof humanWritable !== 'boolean') {
      errors.push(
        `slot "${key}": capabilities.lifecycle.states[].humanWritable must be boolean — skipped`,
      );
      return null;
    }
    const item: SlotState = { value, label, humanWritable };
    if (s.role !== undefined) {
      if (typeof s.role !== 'string' || !STATE_ROLES.has(s.role)) {
        errors.push(
          `slot "${key}": capabilities.lifecycle.states[].role is invalid — skipped`,
        );
        return null;
      }
      item.role = s.role as SlotState['role'];
    }
    if (s.dim !== undefined) {
      if (typeof s.dim !== 'boolean') {
        errors.push(
          `slot "${key}": capabilities.lifecycle.states[].dim must be boolean — skipped`,
        );
        return null;
      }
      item.dim = s.dim;
    }
    if (s.badge !== undefined) {
      if (!isLabel(s.badge)) {
        errors.push(
          `slot "${key}": capabilities.lifecycle.states[].badge is invalid — skipped`,
        );
        return null;
      }
      item.badge = s.badge;
    }
    if (s.aliases !== undefined) {
      if (
        !Array.isArray(s.aliases) ||
        s.aliases.length > LIMITS.stateAliases ||
        !s.aliases.every(
          (a) =>
            typeof a === 'string' &&
            IDENTIFIER_RE.test(a) &&
            a.length <= LIMITS.identifierChars,
        )
      ) {
        errors.push(
          `slot "${key}": capabilities.lifecycle.states[].aliases must be a string array of valid identifiers — skipped`,
        );
        return null;
      }
      item.aliases = s.aliases as string[];
    }
    out.push(item);
  }
  return out;
}

export function parseLifecycle(
  raw: unknown,
  key: string,
  errors: string[],
): LifecycleCapability | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": capabilities.lifecycle must be an object — skipped`);
    return null;
  }
  if (!isWireField(raw.statusField)) {
    errors.push(
      `slot "${key}": capabilities.lifecycle.statusField is required — skipped`,
    );
    return null;
  }
  if (raw.verifiedField !== undefined && !isWireField(raw.verifiedField)) {
    errors.push(
      `slot "${key}": capabilities.lifecycle.verifiedField is invalid — skipped`,
    );
    return null;
  }
  if (raw.rejectionReasonField !== undefined && !isWireField(raw.rejectionReasonField)) {
    errors.push(
      `slot "${key}": capabilities.lifecycle.rejectionReasonField is invalid — skipped`,
    );
    return null;
  }
  if (raw.labelSourceField !== undefined && !isWireField(raw.labelSourceField)) {
    errors.push(
      `slot "${key}": capabilities.lifecycle.labelSourceField is invalid — skipped`,
    );
    return null;
  }
  const states = parseStates(raw.states, key, errors);
  if (states === null) return null;
  const values = new Set(states.map((s) => s.value));
  if (typeof raw.confirmState !== 'string' || !values.has(raw.confirmState)) {
    errors.push(
      `slot "${key}": capabilities.lifecycle.confirmState must name a states[].value — skipped`,
    );
    return null;
  }
  if (typeof raw.rejectState !== 'string' || !values.has(raw.rejectState)) {
    errors.push(
      `slot "${key}": capabilities.lifecycle.rejectState must name a states[].value — skipped`,
    );
    return null;
  }
  if (
    raw.falsePositiveState !== undefined &&
    (typeof raw.falsePositiveState !== 'string' || !values.has(raw.falsePositiveState))
  ) {
    errors.push(
      `slot "${key}": capabilities.lifecycle.falsePositiveState must name a states[].value — skipped`,
    );
    return null;
  }
  const out: LifecycleCapability = {
    statusField: raw.statusField,
    states,
    confirmState: raw.confirmState,
    rejectState: raw.rejectState,
  };
  if (raw.verifiedField !== undefined) out.verifiedField = raw.verifiedField as string;
  if (raw.rejectionReasonField !== undefined)
    out.rejectionReasonField = raw.rejectionReasonField as string;
  if (raw.labelSourceField !== undefined)
    out.labelSourceField = raw.labelSourceField as string;
  if (raw.falsePositiveState !== undefined)
    out.falsePositiveState = raw.falsePositiveState as string;
  return out;
}
