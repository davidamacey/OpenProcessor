/**
 * Tier-2 deployment slot-config parser/validator
 * (the annotation slots contract draft §4/§5, and the tier-2
 * annotation profile config plan §4.2).
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

function resolveContext(ctxIn: ParseContext): Required<ParseContext> {
  const seeded = defaultParseContext();
  return {
    claimedCombos: ctxIn.claimedCombos ?? seeded.claimedCombos,
    takenUrlIds: ctxIn.takenUrlIds ?? seeded.takenUrlIds,
    takenEndpointIds: ctxIn.takenEndpointIds ?? seeded.takenEndpointIds,
    takenKeys: ctxIn.takenKeys ?? seeded.takenKeys,
  };
}

/** Parse every capability; null means the slot is rejected (error already pushed). */
function parseCapabilities(
  raw: unknown,
  key: string,
  errors: string[],
  ctx: Required<ParseContext>,
): SlotSpec['capabilities'] | null {
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": capabilities must be an object — skipped`);
    return null;
  }
  for (const k of Object.keys(raw)) {
    if (!CAPABILITY_KEYS.has(k)) {
      errors.push(`slot "${key}": capabilities has an unknown key "${k}" — ignored`);
    }
  }
  const parsers: [keyof SlotSpec['capabilities'], () => unknown][] = [
    ['subBox', () => parseSubBox(raw.subBox, key, errors)],
    ['text', () => parseText(raw.text, key, errors)],
    ['provenance', () => parseProvenance(raw.provenance, key, errors)],
    ['lifecycle', () => parseLifecycle(raw.lifecycle, key, errors)],
    ['queue', () => parseQueue(raw.queue, key, errors, ctx)],
    ['trainingCohorts', () => parseTrainingCohorts(raw.trainingCohorts, key, errors)],
  ];
  const caps: Record<string, unknown> = {};
  for (const [name, parse] of parsers) {
    const parsed = parse();
    if (parsed === null) return null;
    if (parsed !== undefined) caps[name] = parsed;
  }
  return caps as SlotSpec['capabilities'];
}

/** Slot key, or an error message when the top level is unusable. */
function topLevelProblem(raw: unknown): string | null {
  if (!isPlainObject(raw)) return 'slot spec is not a valid object — skipped';
  // Only a SHALLOW forbidden-key check at the top level: each recognized
  // sub-field below runs its own full recursive `hasForbiddenKey`, which
  // reports pollution under the field that actually carries it. Keys under
  // fields this parser never reads are inert: never copied to the output.
  if (Object.keys(raw).some((k) => FORBIDDEN_KEYS.has(k))) {
    return 'slot spec has a forbidden top-level key — skipped';
  }
  if (!isIdentifier(raw.key)) return 'slot spec has an invalid `key` — skipped';
  return null;
}

export function parseSlotConfig(raw: unknown, ctxIn: ParseContext = {}): SlotParseResult {
  const ctx = resolveContext(ctxIn);
  const errors: string[] = [];
  const problem = topLevelProblem(raw);
  if (problem) {
    errors.push(problem);
    return { errors };
  }
  const rec = raw as Record<string, unknown>;
  const key = rec.key as string;

  if (ctx.takenKeys.has(key)) {
    errors.push(
      `slot "${key}": overrides a built-in slot key — ignored (merge-by-replace)`,
    );
  }

  const bind = parseBind(rec.bind, key, errors);
  if (!bind) return { errors };
  const label = parseLabel(rec.label, key, errors);
  if (!label) return { errors };
  const capabilities = parseCapabilities(rec.capabilities, key, errors, ctx);
  if (capabilities === null) return { errors };
  const endpoints = parseEndpoints(rec.endpoints, key, errors);
  if (endpoints === null) return { errors };
  const stats = parseStats(rec.stats, key, errors);
  if (stats === null) return { errors };
  const extras = parseExtras(rec.extras, key, errors);
  if (extras === null) return { errors };

  const slot: SlotSpec = { key, bind, label, capabilities, endpoints };
  if (stats !== undefined) slot.stats = stats;
  if (extras !== undefined) slot.extras = extras;
  ctx.takenKeys.add(key);
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
