/** `capabilities.queue` parser: ids, keymap and text filter. */

import type { SlotAction, QueueCapability } from '../types';
import {
  LIMITS,
  KEY_COMBO_VOCABULARY,
  FORBIDDEN_SLOT_COMBOS,
  KEYMAP_ACTIONS,
} from './allowLists';
import { validatePathTemplate } from './templatePath';
import {
  hasForbiddenKey,
  isIdentifier,
  isLabel,
  isPlainObject,
  type ParseContext,
} from './parseCommon';

/* ------------------------------------------------------------------ */
/* queue                                                               */
/* ------------------------------------------------------------------ */

/** Why a combo list is unusable for `action` (message suffix after `keymap.`), or null. */
function comboListProblem(action: string, combosRaw: unknown): string | null {
  if (
    !Array.isArray(combosRaw) ||
    combosRaw.length === 0 ||
    combosRaw.length > LIMITS.combosPerAction
  ) {
    return `${action} must be a non-empty array ≤ ${LIMITS.combosPerAction}`;
  }
  for (const c of combosRaw as unknown[]) {
    if (typeof c !== 'string' || !KEY_COMBO_VOCABULARY.has(c)) {
      return `${action} has an unrecognized combo "${String(c)}"`;
    }
  }
  return null;
}

/** Why a non-confirm action cannot claim `combos`, or null; records claims on success. */
function claimProblem(
  action: string,
  combos: string[],
  claimed: Map<string, string>,
  localClaims: Map<string, string>,
): string | null {
  for (const c of combos) {
    if (FORBIDDEN_SLOT_COMBOS.has(c)) {
      return `${action} claims "${c}", which is reserved by the global review-page bindings`;
    }
    const claimedBy = claimed.get(c) ?? localClaims.get(c);
    if (claimedBy !== undefined && claimedBy !== action) {
      return `${action} claims "${c}", already bound to "${claimedBy}"`;
    }
    localClaims.set(c, action);
  }
  return null;
}

function parseKeymap(
  raw: unknown,
  key: string,
  errors: string[],
  claimed: Map<string, string>,
): Partial<Record<SlotAction, string[]>> | null {
  const fail = (msg: string): null => {
    errors.push(`slot "${key}": capabilities.queue.keymap${msg} — skipped`);
    return null;
  };
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) return fail(' must be an object');
  const out: Partial<Record<SlotAction, string[]>> = {};
  const localClaims = new Map<string, string>();
  for (const [action, combosRaw] of Object.entries(raw)) {
    if (!KEYMAP_ACTIONS.includes(action)) {
      return fail(` has an unknown action "${action}"`);
    }
    const listProblem = comboListProblem(action, combosRaw);
    if (listProblem) return fail(`.${listProblem}`);
    const combos = combosRaw as string[];
    if (action === 'confirm') {
      if (combos.length !== 1 || combos[0] !== 'enter') {
        return fail('.confirm is always "enter" and may not be reassigned');
      }
    } else {
      const problem = claimProblem(action, combos, claimed, localClaims);
      if (problem) return fail(`.${problem}`);
    }
    out[action as SlotAction] = combos;
  }
  for (const [combo, action] of localClaims) claimed.set(combo, action);
  return out;
}

function parseTextFilter(
  raw: unknown,
  key: string,
  errors: string[],
): QueueCapability['textFilter'] | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(
      `slot "${key}": capabilities.queue.textFilter must be an object — skipped`,
    );
    return null;
  }
  const { param, label, placeholder } = raw;
  if (!isIdentifier(param)) {
    errors.push(
      `slot "${key}": capabilities.queue.textFilter.param is invalid — skipped`,
    );
    return null;
  }
  if (!isLabel(label)) {
    errors.push(
      `slot "${key}": capabilities.queue.textFilter.label is invalid — skipped`,
    );
    return null;
  }
  if (typeof placeholder !== 'string' || placeholder.length > LIMITS.labelChars) {
    errors.push(
      `slot "${key}": capabilities.queue.textFilter.placeholder is invalid — skipped`,
    );
    return null;
  }
  return { param, label, placeholder };
}

/** Identifier that must not already be taken; null when fine (message suffix). */
function idProblem(
  field: 'endpointId' | 'urlId',
  v: unknown,
  taken: Set<string> | undefined,
): string | null {
  if (!isIdentifier(v)) return `${field} is invalid`;
  return taken?.has(v) ? `${field} "${v}" is already in use` : null;
}

export function parseQueue(
  raw: unknown,
  key: string,
  errors: string[],
  ctx: ParseContext,
): QueueCapability | null | undefined {
  if (raw === undefined) return undefined;
  const fail = (msg: string): null => {
    errors.push(`slot "${key}": capabilities.queue${msg} — skipped`);
    return null;
  };
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) return fail(' must be an object');
  const endpointProblem = idProblem('endpointId', raw.endpointId, ctx.takenEndpointIds);
  if (endpointProblem) return fail(`.${endpointProblem}`);
  const urlProblem = idProblem('urlId', raw.urlId, ctx.takenUrlIds);
  if (urlProblem) return fail(`.${urlProblem}`);
  if (!isLabel(raw.tabLabel)) return fail('.tabLabel is invalid');
  const browsePathResult = validatePathTemplate(raw.browsePath, []);
  if (!browsePathResult.template) return fail(`.browsePath — ${browsePathResult.error}`);
  const claimed = ctx.claimedCombos ?? new Map<string, string>();
  const keymap = parseKeymap(raw.keymap, key, errors, claimed);
  if (keymap === null) return null;
  const textFilter = parseTextFilter(raw.textFilter, key, errors);
  if (textFilter === null) return null;
  if (typeof raw.alwaysVisible !== 'boolean') {
    return fail('.alwaysVisible is required and must be boolean');
  }

  const endpointId = raw.endpointId as string;
  const urlId = raw.urlId as string;
  ctx.takenEndpointIds?.add(endpointId);
  ctx.takenUrlIds?.add(urlId);

  const out: QueueCapability = {
    endpointId,
    urlId,
    tabLabel: raw.tabLabel as string,
    browsePath: raw.browsePath as string,
    keymap,
    alwaysVisible: raw.alwaysVisible,
  };
  if (textFilter !== undefined) out.textFilter = textFilter;
  return out;
}
