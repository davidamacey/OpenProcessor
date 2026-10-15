/**
 * Tier-2 deployment slot-config parser/validator
 * (docs/annotation-slots-contract-draft.md §4/§5, this repo's
 * docs/design/tier2-annotation-profile-config-plan-2026-09-20.md §4.2).
 *
 * `static/annotation-profiles.json` is operator-supplied, untrusted
 * input at a real system boundary — NOT developer-reviewed-PR-supplied
 * code. Every output `SlotSpec` field is assigned individually from a
 * validated primitive; this module never spreads or `Object.assign`s
 * raw input and never round-trips through `JSON.stringify`/`JSON.parse`
 * as a construction shortcut, which is what makes the
 * prototype-pollution and unknown-key guarantees true rather than
 * aspirational.
 *
 * A rejection pushes `slot "<key>": <what> — skipped` (the whole slot is
 * dropped, everything else survives). A soft finding pushes
 * `slot "<key>": <what> — ignored` and parsing continues.
 */

import type { SlotSpec, SlotEndpoints } from '../types';
import { LIMITS, FORBIDDEN_KEYS } from './allowLists';
import { validatePathTemplate, renderPathTemplate } from './templatePath';
import { builtinSlots } from '../registeredSlots';
import {
  hasForbiddenKey,
  isIdentifier,
  isLabel,
  isPlainObject,
  parseExtras,
  type ParseContext,
} from './parseCommon';
import { parseSubBox } from './parseSlotRegion';
import { parseProvenance, parseText } from './parseSlotAppearance';
import { parseLifecycle } from './parseSlotLifecycle';
import { parseQueue } from './parseSlotQueue';
import { parseTrainingCohorts } from './parseSlotCohorts';

const CAPABILITY_KEYS: ReadonlySet<string> = new Set([
  'subBox',
  'text',
  'provenance',
  'lifecycle',
  'queue',
  'trainingCohorts',
]);
export interface SlotParseResult {
  slot?: SlotSpec;
  errors: string[];
}

export interface ProfileDocumentResult {
  slots: SlotSpec[];
  warnings: string[];
}

/* ------------------------------------------------------------------ */
/* bind / label                                                        */
/* ------------------------------------------------------------------ */

function parseBind(
  raw: unknown,
  key: string,
  errors: string[],
): SlotSpec['bind'] | undefined {
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": bind must be an object — skipped`);
    return undefined;
  }
  const out: { className?: string; classId?: number } = {};
  let has = false;
  if (raw.className !== undefined) {
    if (!isLabel(raw.className)) {
      errors.push(`slot "${key}": bind.className must be a non-empty string — skipped`);
      return undefined;
    }
    out.className = raw.className;
    has = true;
  }
  if (raw.classId !== undefined) {
    if (
      typeof raw.classId !== 'number' ||
      !Number.isInteger(raw.classId) ||
      raw.classId < 0
    ) {
      errors.push(`slot "${key}": bind.classId must be a non-negative integer — skipped`);
      return undefined;
    }
    out.classId = raw.classId;
    has = true;
  }
  if (!has) {
    errors.push(`slot "${key}": bind must include className or classId — skipped`);
    return undefined;
  }
  return out;
}

function parseLabel(
  raw: unknown,
  key: string,
  errors: string[],
): SlotSpec['label'] | undefined {
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": label must be an object — skipped`);
    return undefined;
  }
  const { singular, plural, title } = raw;
  if (!isLabel(singular) || !isLabel(plural) || !isLabel(title)) {
    errors.push(
      `slot "${key}": label.singular/.plural/.title are all required non-empty strings — skipped`,
    );
    return undefined;
  }
  return { singular, plural, title };
}
/* ------------------------------------------------------------------ */
/* endpoints / stats                                                   */
/* ------------------------------------------------------------------ */

function parseCropIdEndpoint(
  raw: unknown,
  field: string,
  key: string,
  errors: string[],
): ((cropId: string) => string) | null | undefined {
  if (raw === undefined) return undefined;
  const result = validatePathTemplate(raw, ['cropId']);
  if (
    !result.template ||
    !result.placeholders ||
    result.placeholders.length !== 1 ||
    result.placeholders[0] !== 'cropId'
  ) {
    errors.push(
      `slot "${key}": endpoints.${field} must use {cropId} and nothing else — skipped`,
    );
    return null;
  }
  const template = result.template;
  return (cropId: string) => renderPathTemplate(template, { cropId });
}

function parseEndpoints(
  raw: unknown,
  key: string,
  errors: string[],
): SlotEndpoints | null {
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": endpoints must be an object — skipped`);
    return null;
  }
  const out: SlotEndpoints = {};
  for (const field of ['patchMeta'] as const) {
    const fn = parseCropIdEndpoint(raw[field], field, key, errors);
    if (fn === null) return null;
    if (fn !== undefined) out[field] = fn;
  }
  if (raw.batchStatus !== undefined) {
    const result = validatePathTemplate(raw.batchStatus, []);
    if (!result.template) {
      errors.push(`slot "${key}": endpoints.batchStatus — ${result.error} — skipped`);
      return null;
    }
    const template = result.template;
    out.batchStatus = () => template;
  }
  return out;
}

function parseStats(
  raw: unknown,
  key: string,
  errors: string[],
): SlotSpec['stats'] | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": stats must be an object — skipped`);
    return null;
  }
  if (!isIdentifier(raw.key)) {
    errors.push(`slot "${key}": stats.key is invalid — skipped`);
    return null;
  }
  if (!isLabel(raw.panelTitle)) {
    errors.push(`slot "${key}": stats.panelTitle is invalid — skipped`);
    return null;
  }
  if (!isLabel(raw.coverageTitle)) {
    errors.push(`slot "${key}": stats.coverageTitle is invalid — skipped`);
    return null;
  }
  return { key: raw.key, panelTitle: raw.panelTitle, coverageTitle: raw.coverageTitle };
}

/* ------------------------------------------------------------------ */
/* Slot / document entry points                                        */
/* ------------------------------------------------------------------ */

/**
 * The default for any `ParseContext` field the caller does not supply, so
 * a bare `parseSlotConfig(raw)` call (no ctx, e.g. every unit test that
 * isn't specifically exercising cross-slot collision) still rejects a
 * `urlId`/`endpointId`/`key`/keymap combo that would collide with a real
 * deployment's resolved state.
 *
 * Slot keys and keymap combos come from the build's own tier-1 slots
 * (`builtinSlots`), read at call time. The review tab and preset ids are
 * CORE_REVIEW_TABS + REVIEW_PRESETS (../../reviewTabs.ts), kept as a
 * literal rather than importing that module, to avoid this parser
 * depending on the application wiring it sits in front of.
 */
function defaultParseContext(): Required<ParseContext> {
  const claimedCombos = new Map<string, string>();
  const takenKeys = new Set<string>();
  for (const slot of builtinSlots) {
    takenKeys.add(slot.key);
    const keymap: Record<string, readonly string[] | undefined> =
      slot.capabilities.queue?.keymap ?? {};
    for (const [action, combos] of Object.entries(keymap)) {
      for (const combo of combos ?? []) {
        if (!claimedCombos.has(combo)) claimedCombos.set(combo, action);
      }
    }
  }
  return {
    claimedCombos,
    takenUrlIds: new Set([
      'all',
      'uncertainty',
      'model_disagreements',
      'classifier_blind_spots',
      'mismatches',
      'vlm_low_conf',
      'primary_low_conf',
    ]),
    takenEndpointIds: new Set([
      'all',
      'uncertainty',
      'model_disagreements',
      'classifier_blind_spots',
    ]),
    takenKeys,
  };
}

export function parseSlotConfig(raw: unknown, ctxIn: ParseContext = {}): SlotParseResult {
  const seeded = defaultParseContext();
  const ctx: Required<ParseContext> = {
    claimedCombos: ctxIn.claimedCombos ?? seeded.claimedCombos,
    takenUrlIds: ctxIn.takenUrlIds ?? seeded.takenUrlIds,
    takenEndpointIds: ctxIn.takenEndpointIds ?? seeded.takenEndpointIds,
    takenKeys: ctxIn.takenKeys ?? seeded.takenKeys,
  };
  const errors: string[] = [];
  if (!isPlainObject(raw)) {
    errors.push('slot spec is not a valid object — skipped');
    return { errors };
  }
  // Only a SHALLOW forbidden-key check at the top level: each recognized
  // sub-field (bind/label/capabilities/extras/endpoints/stats) below runs
  // its own full recursive `hasForbiddenKey`, which both catches pollution
  // in the field that actually carries it AND reports it with a message
  // naming that field. A deep top-level scan here would instead reject
  // for "slot spec is not a valid object" regardless of which field was
  // actually poisoned — true but useless to an operator debugging their
  // JSON. Any forbidden key nested under a field this parser doesn't read
  // at all is inert: it is never copied into the output `SlotSpec`.
  if (Object.keys(raw).some((k) => FORBIDDEN_KEYS.has(k))) {
    errors.push('slot spec has a forbidden top-level key — skipped');
    return { errors };
  }
  if (!isIdentifier(raw.key)) {
    errors.push('slot spec has an invalid `key` — skipped');
    return { errors };
  }
  const key = raw.key;

  if (ctx.takenKeys?.has(key)) {
    errors.push(
      `slot "${key}": overrides a built-in slot key — ignored (merge-by-replace)`,
    );
  }

  const bind = parseBind(raw.bind, key, errors);
  if (!bind) return { errors };

  const label = parseLabel(raw.label, key, errors);
  if (!label) return { errors };

  if (!isPlainObject(raw.capabilities) || hasForbiddenKey(raw.capabilities)) {
    errors.push(`slot "${key}": capabilities must be an object — skipped`);
    return { errors };
  }
  const capsRaw = raw.capabilities;
  for (const k of Object.keys(capsRaw)) {
    if (!CAPABILITY_KEYS.has(k)) {
      errors.push(`slot "${key}": capabilities has an unknown key "${k}" — ignored`);
    }
  }

  const subBox = parseSubBox(capsRaw.subBox, key, errors);
  if (subBox === null) return { errors };
  const text = parseText(capsRaw.text, key, errors);
  if (text === null) return { errors };
  const provenance = parseProvenance(capsRaw.provenance, key, errors);
  if (provenance === null) return { errors };
  const lifecycle = parseLifecycle(capsRaw.lifecycle, key, errors);
  if (lifecycle === null) return { errors };
  const queue = parseQueue(capsRaw.queue, key, errors, ctx);
  if (queue === null) return { errors };
  const trainingCohorts = parseTrainingCohorts(capsRaw.trainingCohorts, key, errors);
  if (trainingCohorts === null) return { errors };

  const endpoints = parseEndpoints(raw.endpoints, key, errors);
  if (endpoints === null) return { errors };

  const stats = parseStats(raw.stats, key, errors);
  if (stats === null) return { errors };

  const extras = parseExtras(raw.extras, key, errors);
  if (extras === null) return { errors };

  const slot: SlotSpec = { key, bind, label, capabilities: {}, endpoints };
  if (subBox !== undefined) slot.capabilities.subBox = subBox;
  if (text !== undefined) slot.capabilities.text = text;
  if (provenance !== undefined) slot.capabilities.provenance = provenance;
  if (lifecycle !== undefined) slot.capabilities.lifecycle = lifecycle;
  if (queue !== undefined) slot.capabilities.queue = queue;
  if (trainingCohorts !== undefined) slot.capabilities.trainingCohorts = trainingCohorts;
  if (stats !== undefined) slot.stats = stats;
  if (extras !== undefined) slot.extras = extras;

  ctx.takenKeys?.add(key);

  return { slot, errors };
}

export function parseProfileDocument(raw: unknown): ProfileDocumentResult {
  const warnings: string[] = [];
  let slotsRaw: unknown[];

  if (Array.isArray(raw)) {
    slotsRaw = raw;
  } else if (isPlainObject(raw)) {
    if (hasForbiddenKey(raw)) {
      warnings.push('document contains a forbidden key — rejected');
      return { slots: [], warnings };
    }
    if (raw.version !== 1) {
      warnings.push('document has an unrecognized `version` (expected 1) — rejected');
      return { slots: [], warnings };
    }
    if (!Array.isArray(raw.slots)) {
      warnings.push('document is missing a `slots` array — rejected');
      return { slots: [], warnings };
    }
    slotsRaw = raw.slots;
  } else {
    warnings.push(
      'document must be an object with a `slots` array, or a bare array — rejected',
    );
    return { slots: [], warnings };
  }

  if (slotsRaw.length > LIMITS.slots) {
    warnings.push(`document declares more than ${LIMITS.slots} slots — rejected`);
    return { slots: [], warnings };
  }

  const ctx: ParseContext = defaultParseContext();

  const slots: SlotSpec[] = [];
  for (const candidate of slotsRaw) {
    const { slot, errors } = parseSlotConfig(candidate, ctx);
    warnings.push(...errors);
    if (slot) slots.push(slot);
  }
  return { slots, warnings };
}
