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

function parseKeymap(
  raw: unknown,
  key: string,
  errors: string[],
  claimed: Map<string, string>,
): Partial<Record<SlotAction, string[]>> | null {
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": capabilities.queue.keymap must be an object — skipped`);
    return null;
  }
  const out: Partial<Record<SlotAction, string[]>> = {};
  const localClaims = new Map<string, string>();
  for (const [action, combosRaw] of Object.entries(raw)) {
    if (!KEYMAP_ACTIONS.includes(action)) {
      errors.push(
        `slot "${key}": capabilities.queue.keymap has an unknown action "${action}" — skipped`,
      );
      return null;
    }
    if (
      !Array.isArray(combosRaw) ||
      combosRaw.length === 0 ||
      combosRaw.length > LIMITS.combosPerAction
    ) {
      errors.push(
        `slot "${key}": capabilities.queue.keymap.${action} must be a non-empty array ≤ ${LIMITS.combosPerAction} — skipped`,
      );
      return null;
    }
    const combos: string[] = [];
    for (const c of combosRaw as unknown[]) {
      if (typeof c !== 'string' || !KEY_COMBO_VOCABULARY.has(c)) {
        errors.push(
          `slot "${key}": capabilities.queue.keymap.${action} has an unrecognized combo "${String(c)}" — skipped`,
        );
        return null;
      }
      combos.push(c);
    }
    if (action === 'confirm') {
      if (combos.length !== 1 || combos[0] !== 'enter') {
        errors.push(
          `slot "${key}": capabilities.queue.keymap.confirm is always "enter" and may not be reassigned — skipped`,
        );
        return null;
      }
      out.confirm = combos;
      continue;
    }
    for (const c of combos) {
      if (FORBIDDEN_SLOT_COMBOS.has(c)) {
        errors.push(
          `slot "${key}": capabilities.queue.keymap.${action} claims "${c}", which is reserved by the global review-page bindings — skipped`,
        );
        return null;
      }
      const claimedBy = claimed.get(c) ?? localClaims.get(c);
      if (claimedBy !== undefined && claimedBy !== action) {
        errors.push(
          `slot "${key}": capabilities.queue.keymap.${action} claims "${c}", already bound to "${claimedBy}" — skipped`,
        );
        return null;
      }
      localClaims.set(c, action);
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

export function parseQueue(
  raw: unknown,
  key: string,
  errors: string[],
  ctx: ParseContext,
): QueueCapability | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": capabilities.queue must be an object — skipped`);
    return null;
  }
  const endpointId = raw.endpointId;
  if (!isIdentifier(endpointId)) {
    errors.push(`slot "${key}": capabilities.queue.endpointId is invalid — skipped`);
    return null;
  }
  if (ctx.takenEndpointIds?.has(endpointId)) {
    errors.push(
      `slot "${key}": capabilities.queue.endpointId "${endpointId}" is already in use — skipped`,
    );
    return null;
  }
  const urlId = raw.urlId;
  if (!isIdentifier(urlId)) {
    errors.push(`slot "${key}": capabilities.queue.urlId is invalid — skipped`);
    return null;
  }
  if (ctx.takenUrlIds?.has(urlId)) {
    errors.push(
      `slot "${key}": capabilities.queue.urlId "${urlId}" is already in use — skipped`,
    );
    return null;
  }
  if (!isLabel(raw.tabLabel)) {
    errors.push(`slot "${key}": capabilities.queue.tabLabel is invalid — skipped`);
    return null;
  }
  const browsePathResult = validatePathTemplate(raw.browsePath, []);
  if (!browsePathResult.template) {
    errors.push(
      `slot "${key}": capabilities.queue.browsePath — ${browsePathResult.error} — skipped`,
    );
    return null;
  }
  const claimed = ctx.claimedCombos ?? new Map<string, string>();
  const keymap = parseKeymap(raw.keymap, key, errors, claimed);
  if (keymap === null) return null;
  const textFilter = parseTextFilter(raw.textFilter, key, errors);
  if (textFilter === null) return null;
  if (typeof raw.alwaysVisible !== 'boolean') {
    errors.push(
      `slot "${key}": capabilities.queue.alwaysVisible is required and must be boolean — skipped`,
    );
    return null;
  }

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
