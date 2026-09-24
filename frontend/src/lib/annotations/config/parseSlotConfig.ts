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

import type {
  SlotSpec,
  SlotAction,
  SubBoxCapability,
  TextCapability,
  ProvenanceCapability,
  LifecycleCapability,
  QueueCapability,
  SlotEndpoints,
  SlotState,
  ShapeEnvelope,
  SubBoxRing,
} from '../types';
import type {
  CohortSpec,
  CohortQuery,
  CohortFilter,
  CohortOp,
  TrainingCohortsCapability,
} from '../cohorts';
import {
  LIMITS,
  IDENTIFIER_RE,
  WIRE_FIELD_RE,
  FORBIDDEN_KEYS,
  RING_PRESETS,
  RING_CLASS_ALLOWLIST,
  KEY_COMBO_VOCABULARY,
  FORBIDDEN_SLOT_COMBOS,
  KEYMAP_ACTIONS,
  REGEX_FLAG_ALLOWLIST,
  PATH_PLACEHOLDERS,
  COHORT_PLACEHOLDERS,
} from './allowLists';
import { validatePathTemplate, renderPathTemplate } from './templatePath';

/** Conservative ReDoS shape check. Catches the catastrophic-backtracking
 *  family `(x+)+`, `(x*)*`, `(x+)*`, `(x{n,})+` — a group whose body ends
 *  in an unbounded quantifier and which is itself quantified. It is a
 *  HEURISTIC, not a proof: the real bounds are `patternSourceChars` and
 *  `maxLength`, which cap both the pattern and the input it runs on. */
const NESTED_QUANTIFIER = /\([^()]*[+*]\s*(\{\d+,\d*\})?\)\s*[+*{]/;

const STATE_ROLES: ReadonlySet<string> = new Set([
  'proposed',
  'confirmed',
  'rejected',
  'falsePositive',
  'absent',
  'pending',
]);

const CAPABILITY_KEYS: ReadonlySet<string> = new Set([
  'subBox',
  'text',
  'provenance',
  'lifecycle',
  'queue',
  'trainingCohorts',
]);

const COHORT_OPS: ReadonlySet<string> = new Set(['exists', 'eq', 'lt', 'containsAll']);

export interface SlotParseResult {
  slot?: SlotSpec;
  errors: string[];
}

export interface ParseContext {
  /** Combos already claimed by a previously-resolved slot, mapped to
   *  the action that claimed them. Cross-slot collisions on a DIFFERENT
   *  action reject; the same action is fine. */
  claimedCombos?: Map<string, string>;
  /** `?tab=` values already in use — core review tabs and preset ids as
   *  well as earlier slots. */
  takenUrlIds?: Set<string>;
  takenEndpointIds?: Set<string>;
  takenKeys?: Set<string>;
}

export interface ProfileDocumentResult {
  slots: SlotSpec[];
  warnings: string[];
}

/* ------------------------------------------------------------------ */
/* Generic helpers                                                     */
/* ------------------------------------------------------------------ */

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function hasForbiddenKey(v: unknown): boolean {
  if (Array.isArray(v)) return v.some((x) => hasForbiddenKey(x));
  if (isPlainObject(v)) {
    for (const k of Object.keys(v)) {
      if (FORBIDDEN_KEYS.has(k)) return true;
      if (hasForbiddenKey(v[k])) return true;
    }
    return false;
  }
  return false;
}

function isWireField(v: unknown): v is string {
  return (
    typeof v === 'string' &&
    v.length > 0 &&
    v.length <= LIMITS.identifierChars &&
    WIRE_FIELD_RE.test(v)
  );
}

function isIdentifier(v: unknown): v is string {
  return (
    typeof v === 'string' &&
    v.length > 0 &&
    v.length <= LIMITS.identifierChars &&
    IDENTIFIER_RE.test(v)
  );
}

function isLabel(v: unknown): v is string {
  return typeof v === 'string' && v.length > 0 && v.length <= LIMITS.labelChars;
}

/* ------------------------------------------------------------------ */
/* extras                                                              */
/* ------------------------------------------------------------------ */

function depthOf(v: unknown, depth = 0): number {
  if (depth > 8) return depth;
  if (Array.isArray(v)) {
    return v.length ? Math.max(...v.map((x) => depthOf(x, depth + 1))) : depth;
  }
  if (isPlainObject(v)) {
    const keys = Object.keys(v);
    return keys.length ? Math.max(...keys.map((k) => depthOf(v[k], depth + 1))) : depth;
  }
  return depth;
}

/** Builds a fresh object/array field-by-field — deliberately not a
 *  `JSON.parse(JSON.stringify())` shortcut, so the result can never
 *  carry an own `__proto__`/`constructor`/`prototype` key even if a
 *  future caller forgets the earlier `hasForbiddenKey` guard. */
function cloneExtras(v: unknown): unknown {
  if (Array.isArray(v)) return v.map((x) => cloneExtras(x));
  if (isPlainObject(v)) {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(v)) {
      if (FORBIDDEN_KEYS.has(k)) continue;
      out[k] = cloneExtras(v[k]);
    }
    return out;
  }
  return v;
}

function parseExtras(
  raw: unknown,
  key: string,
  errors: string[],
): Record<string, unknown> | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw)) {
    errors.push(`slot "${key}": extras must be a plain object — skipped`);
    return null;
  }
  if (hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": extras contains a forbidden key — skipped`);
    return null;
  }
  if (depthOf(raw) > 8) {
    errors.push(`slot "${key}": extras exceeds the maximum nesting depth of 8 — skipped`);
    return null;
  }
  return cloneExtras(raw) as Record<string, unknown>;
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
/* subBox                                                              */
/* ------------------------------------------------------------------ */

function parseEnvelope(
  raw: unknown,
  key: string,
  errors: string[],
): ShapeEnvelope | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(
      `slot "${key}": capabilities.subBox.envelope must be an object — skipped`,
    );
    return null;
  }
  const out: ShapeEnvelope = {};
  const fields = [
    'aspectMin',
    'aspectMax',
    'maxWidthFrac',
    'maxHeightFrac',
    'maxAreaFrac',
  ] as const;
  for (const f of fields) {
    if (raw[f] === undefined) continue;
    const v = raw[f];
    if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) {
      errors.push(
        `slot "${key}": capabilities.subBox.envelope.${f} must be a positive finite number — skipped`,
      );
      return null;
    }
    out[f] = v;
  }
  if (
    out.aspectMin !== undefined &&
    out.aspectMax !== undefined &&
    out.aspectMin > out.aspectMax
  ) {
    errors.push(
      `slot "${key}": capabilities.subBox.envelope.aspectMin must be <= aspectMax — skipped`,
    );
    return null;
  }
  return out;
}

function parseThumbnail(
  raw: unknown,
  key: string,
  errors: string[],
): SubBoxCapability['thumbnail'] | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(
      `slot "${key}": capabilities.subBox.thumbnail must be an object — skipped`,
    );
    return null;
  }
  const pathResult = validatePathTemplate(raw.path, PATH_PLACEHOLDERS);
  if (!pathResult.template) {
    errors.push(
      `slot "${key}": capabilities.subBox.thumbnail.path — ${pathResult.error} — skipped`,
    );
    return null;
  }
  const aspect = raw.aspect;
  if (typeof aspect !== 'string' || !/^\d{1,3} \/ \d{1,3}$/.test(aspect)) {
    errors.push(
      `slot "${key}": capabilities.subBox.thumbnail.aspect must match "W / H" — skipped`,
    );
    return null;
  }
  const defaultSize = raw.defaultSize;
  if (
    typeof defaultSize !== 'number' ||
    !Number.isInteger(defaultSize) ||
    defaultSize < 16 ||
    defaultSize > 2048
  ) {
    errors.push(
      `slot "${key}": capabilities.subBox.thumbnail.defaultSize must be an integer in [16, 2048] — skipped`,
    );
    return null;
  }
  const template = pathResult.template;
  return {
    path: (cropId: string, size: number) =>
      renderPathTemplate(template, { cropId, size }),
    aspect,
    defaultSize,
  };
}

function parseRing(raw: unknown, key: string, errors: string[]): SubBoxRing | null {
  if (typeof raw === 'string') {
    const preset = RING_PRESETS[raw];
    if (!preset) {
      errors.push(
        `slot "${key}": capabilities.subBox.ring "${raw}" is not a known preset — skipped`,
      );
      return null;
    }
    return preset;
  }
  if (isPlainObject(raw) && !hasForbiddenKey(raw)) {
    const { confirmed, proposed, rejected } = raw;
    if (
      typeof confirmed === 'string' &&
      typeof proposed === 'string' &&
      typeof rejected === 'string' &&
      RING_CLASS_ALLOWLIST.has(confirmed) &&
      RING_CLASS_ALLOWLIST.has(proposed) &&
      RING_CLASS_ALLOWLIST.has(rejected)
    ) {
      return { confirmed, proposed, rejected };
    }
  }
  errors.push(
    `slot "${key}": capabilities.subBox.ring must be a known preset name or an object of three allow-listed class strings — skipped`,
  );
  return null;
}

function parseEditor(
  raw: unknown,
  key: string,
  errors: string[],
): SubBoxCapability['editor'] | null {
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": capabilities.subBox.editor must be an object — skipped`);
    return null;
  }
  const { thumbSize, viewPadding, nudgeStep } = raw;
  if (
    typeof thumbSize !== 'number' ||
    !Number.isInteger(thumbSize) ||
    thumbSize < 64 ||
    thumbSize > 4096
  ) {
    errors.push(
      `slot "${key}": capabilities.subBox.editor.thumbSize must be an integer in [64, 4096] — skipped`,
    );
    return null;
  }
  if (
    typeof viewPadding !== 'number' ||
    !Number.isFinite(viewPadding) ||
    viewPadding <= 0 ||
    viewPadding > 100
  ) {
    errors.push(
      `slot "${key}": capabilities.subBox.editor.viewPadding must be a finite number in (0, 100] — skipped`,
    );
    return null;
  }
  if (
    typeof nudgeStep !== 'number' ||
    !Number.isFinite(nudgeStep) ||
    nudgeStep <= 0 ||
    nudgeStep >= 1
  ) {
    errors.push(
      `slot "${key}": capabilities.subBox.editor.nudgeStep must be a finite number in (0, 1) — skipped`,
    );
    return null;
  }
  return { thumbSize, viewPadding, nudgeStep };
}

function parseSubBox(
  raw: unknown,
  key: string,
  errors: string[],
): SubBoxCapability | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": capabilities.subBox must be an object — skipped`);
    return null;
  }
  if (!isWireField(raw.bboxField)) {
    errors.push(`slot "${key}": capabilities.subBox.bboxField is required — skipped`);
    return null;
  }
  if (raw.storedFrame !== 'source' && raw.storedFrame !== 'parent') {
    errors.push(
      `slot "${key}": capabilities.subBox.storedFrame must be 'source' or 'parent' — skipped`,
    );
    return null;
  }
  for (const f of ['frameField', 'scoreField', 'visibleField'] as const) {
    if (raw[f] !== undefined && !isWireField(raw[f])) {
      errors.push(`slot "${key}": capabilities.subBox.${f} is invalid — skipped`);
      return null;
    }
  }
  const envelope = parseEnvelope(raw.envelope, key, errors);
  if (envelope === null) return null;
  const thumbnail = parseThumbnail(raw.thumbnail, key, errors);
  if (thumbnail === null) return null;
  const ring = parseRing(raw.ring, key, errors);
  if (ring === null) return null;
  const editor = parseEditor(raw.editor, key, errors);
  if (editor === null) return null;

  const out: SubBoxCapability = {
    bboxField: raw.bboxField as string,
    storedFrame: raw.storedFrame,
    ring,
    editor,
  };
  if (raw.frameField !== undefined) out.frameField = raw.frameField as string;
  if (raw.scoreField !== undefined) out.scoreField = raw.scoreField as string;
  if (raw.visibleField !== undefined) out.visibleField = raw.visibleField as string;
  if (envelope !== undefined) out.envelope = envelope;
  if (thumbnail !== undefined) out.thumbnail = thumbnail;
  return out;
}

/* ------------------------------------------------------------------ */
/* text                                                                */
/* ------------------------------------------------------------------ */

function parsePattern(
  raw: unknown,
  key: string,
  errors: string[],
): RegExp | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": capabilities.text.pattern must be an object — skipped`);
    return null;
  }
  const { source } = raw;
  if (
    typeof source !== 'string' ||
    source.length === 0 ||
    source.length > LIMITS.patternSourceChars
  ) {
    errors.push(
      `slot "${key}": capabilities.text.pattern.source must be a non-empty string ≤ ${LIMITS.patternSourceChars} chars — skipped`,
    );
    return null;
  }
  let flags = '';
  if (raw.flags !== undefined) {
    if (typeof raw.flags !== 'string') {
      errors.push(
        `slot "${key}": capabilities.text.pattern.flags must be a string — skipped`,
      );
      return null;
    }
    for (const f of raw.flags) {
      if (!REGEX_FLAG_ALLOWLIST.has(f)) {
        errors.push(
          `slot "${key}": capabilities.text.pattern.flags contains disallowed flag "${f}" — skipped`,
        );
        return null;
      }
    }
    flags = raw.flags;
  }
  if (NESTED_QUANTIFIER.test(source)) {
    errors.push(
      `slot "${key}": capabilities.text.pattern.source rejected by the nested-quantifier ReDoS check — skipped`,
    );
    return null;
  }
  try {
    return new RegExp(source, flags);
  } catch {
    errors.push(
      `slot "${key}": capabilities.text.pattern.source is not a valid regular expression — skipped`,
    );
    return null;
  }
}

function parseVocabulary(
  raw: unknown,
  key: string,
  errors: string[],
): TextCapability['vocabulary'] | null | undefined {
  if (raw === undefined) return undefined;
  if (!Array.isArray(raw) || raw.length > LIMITS.vocabularyEntries) {
    errors.push(
      `slot "${key}": capabilities.text.vocabulary must be an array ≤ ${LIMITS.vocabularyEntries} entries — skipped`,
    );
    return null;
  }
  const seen = new Set<string>();
  const out: NonNullable<TextCapability['vocabulary']> = [];
  for (const entry of raw) {
    if (!isPlainObject(entry) || hasForbiddenKey(entry)) {
      errors.push(
        `slot "${key}": capabilities.text.vocabulary entry must be an object — skipped`,
      );
      return null;
    }
    const { value, label, description } = entry;
    if (!isLabel(value) || !isLabel(label)) {
      errors.push(
        `slot "${key}": capabilities.text.vocabulary entry needs a non-empty value/label — skipped`,
      );
      return null;
    }
    if (description !== undefined && !isLabel(description)) {
      errors.push(
        `slot "${key}": capabilities.text.vocabulary entry description invalid — skipped`,
      );
      return null;
    }
    if (seen.has(value)) {
      errors.push(
        `slot "${key}": capabilities.text.vocabulary has duplicate value "${value}" — skipped`,
      );
      return null;
    }
    seen.add(value);
    const item: { value: string; label: string; description?: string } = { value, label };
    if (description !== undefined) item.description = description;
    out.push(item);
  }
  return out;
}

function parseText(
  raw: unknown,
  key: string,
  errors: string[],
): TextCapability | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": capabilities.text must be an object — skipped`);
    return null;
  }
  if (!isWireField(raw.valueField)) {
    errors.push(`slot "${key}": capabilities.text.valueField is required — skipped`);
    return null;
  }
  for (const f of [
    'rawField',
    'sourceField',
    'confidenceField',
    'engineVersionField',
  ] as const) {
    if (raw[f] !== undefined && !isWireField(raw[f])) {
      errors.push(`slot "${key}": capabilities.text.${f} is invalid — skipped`);
      return null;
    }
  }
  if (!isLabel(raw.label)) {
    errors.push(`slot "${key}": capabilities.text.label is required — skipped`);
    return null;
  }
  if (raw.placeholder !== undefined && !isLabel(raw.placeholder)) {
    errors.push(`slot "${key}": capabilities.text.placeholder is invalid — skipped`);
    return null;
  }
  if (
    raw.transform !== undefined &&
    !['none', 'uppercase', 'lowercase', 'trim'].includes(raw.transform as string)
  ) {
    errors.push(
      `slot "${key}": capabilities.text.transform must be one of none/uppercase/lowercase/trim — skipped`,
    );
    return null;
  }
  const pattern = parsePattern(raw.pattern, key, errors);
  if (pattern === null) return null;
  if (
    raw.maxLength !== undefined &&
    (typeof raw.maxLength !== 'number' ||
      !Number.isInteger(raw.maxLength) ||
      raw.maxLength < 1 ||
      raw.maxLength > 4096)
  ) {
    errors.push(
      `slot "${key}": capabilities.text.maxLength must be an integer in [1, 4096] — skipped`,
    );
    return null;
  }
  if (raw.monospace !== undefined && typeof raw.monospace !== 'boolean') {
    errors.push(`slot "${key}": capabilities.text.monospace must be a boolean — skipped`);
    return null;
  }
  const vocabulary = parseVocabulary(raw.vocabulary, key, errors);
  if (vocabulary === null) return null;

  const out: TextCapability = {
    valueField: raw.valueField as string,
    label: raw.label as string,
  };
  if (raw.rawField !== undefined) out.rawField = raw.rawField as string;
  if (raw.sourceField !== undefined) out.sourceField = raw.sourceField as string;
  if (raw.confidenceField !== undefined)
    out.confidenceField = raw.confidenceField as string;
  if (raw.engineVersionField !== undefined)
    out.engineVersionField = raw.engineVersionField as string;
  if (raw.placeholder !== undefined) out.placeholder = raw.placeholder as string;
  if (raw.transform !== undefined)
    out.transform = raw.transform as TextCapability['transform'];
  if (pattern !== undefined) out.pattern = pattern;
  if (raw.maxLength !== undefined) out.maxLength = raw.maxLength as number;
  if (raw.monospace !== undefined) out.monospace = raw.monospace as boolean;
  if (vocabulary !== undefined) out.vocabulary = vocabulary;
  return out;
}

/* ------------------------------------------------------------------ */
/* provenance                                                          */
/* ------------------------------------------------------------------ */

const PROVENANCE_OPTIONAL = [
  'detectorVersionField',
  'chainField',
  'verifierField',
  'verifierVersionField',
  'verifiedAtField',
  'detectedAtField',
] as const;

function parseProvenance(
  raw: unknown,
  key: string,
  errors: string[],
): ProvenanceCapability | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": capabilities.provenance must be an object — skipped`);
    return null;
  }
  if (!isWireField(raw.detectorField)) {
    errors.push(
      `slot "${key}": capabilities.provenance.detectorField is required — skipped`,
    );
    return null;
  }
  for (const f of PROVENANCE_OPTIONAL) {
    if (raw[f] !== undefined && !isWireField(raw[f])) {
      errors.push(`slot "${key}": capabilities.provenance.${f} is invalid — skipped`);
      return null;
    }
  }
  if (typeof raw.showChainOnCard !== 'boolean') {
    errors.push(
      `slot "${key}": capabilities.provenance.showChainOnCard is required and must be boolean — skipped`,
    );
    return null;
  }
  const out: ProvenanceCapability = {
    detectorField: raw.detectorField as string,
    showChainOnCard: raw.showChainOnCard,
  };
  for (const f of PROVENANCE_OPTIONAL) {
    if (raw[f] !== undefined) out[f] = raw[f] as string;
  }
  return out;
}

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

function parseLifecycle(
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

function parseQueue(
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

function parseTrainingCohorts(
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
  for (const field of ['setBox', 'clearBox', 'patchMeta'] as const) {
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
 * Seeded from CORE_REVIEW_TABS + REVIEW_PRESETS (../../reviewTabs.ts) and
 * licensePlateSlot's keymap (../profiles/licensePlate.ts) — kept as a
 * literal here rather than importing those modules, to avoid this parser
 * depending on application wiring it is meant to sit in front of. Used as
 * the default for any `ParseContext` field the caller does not supply —
 * a bare `parseSlotConfig(raw)` call (no ctx, e.g. every unit test that
 * isn't specifically exercising cross-slot collision) still rejects a
 * `urlId`/`endpointId`/`key`/keymap combo that would collide with a real
 * deployment's resolved state.
 */
function defaultParseContext(): Required<ParseContext> {
  return {
    claimedCombos: new Map([
      ['d', 'reject'],
      ['f', 'markFalsePositive'],
      ['e', 'editBox'],
      ['b', 'back'],
      ['arrowleft', 'back'],
    ]),
    takenUrlIds: new Set([
      'all',
      'uncertainty',
      'model_disagreements',
      'coco_blind_spots',
      'mismatches',
      'vlm_low_conf',
      'primary_low_conf',
    ]),
    takenEndpointIds: new Set([
      'all',
      'uncertainty',
      'model_disagreements',
      'coco_blind_spots',
    ]),
    takenKeys: new Set(['license_plate']),
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
