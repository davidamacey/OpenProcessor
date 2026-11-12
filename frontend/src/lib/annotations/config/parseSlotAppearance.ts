/** `capabilities.text` and `capabilities.provenance` parsers. */

import type { TextCapability, ProvenanceCapability } from '../types';
import { LIMITS, REGEX_FLAG_ALLOWLIST } from './allowLists';
import { hasForbiddenKey, isLabel, isPlainObject, isWireField } from './parseCommon';

/** Conservative ReDoS shape check. Catches the catastrophic-backtracking
 *  family `(x+)+`, `(x*)*`, `(x+)*`, `(x{n,})+` — a group whose body ends
 *  in an unbounded quantifier and which is itself quantified. It is a
 *  HEURISTIC, not a proof: the real bounds are `patternSourceChars` and
 *  `maxLength`, which cap both the pattern and the input it runs on. */
const NESTED_QUANTIFIER = /\([^()]*[+*]\s*(\{\d+,\d*\})?\)\s*[+*{]/;
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

const TEXT_WIRE_FIELDS = [
  'valueField',
  'rawField',
  'sourceField',
  'confidenceField',
  'engineVersionField',
] as const;
const TEXT_TRANSFORMS: readonly unknown[] = ['none', 'uppercase', 'lowercase', 'trim'];

function isValidMaxLength(v: unknown): boolean {
  return typeof v === 'number' && Number.isInteger(v) && v >= 1 && v <= 4096;
}

/** First scalar-field problem of a text capability, or null. */
function textScalarProblem(raw: Record<string, unknown>): string | null {
  for (const f of TEXT_WIRE_FIELDS) {
    if (raw[f] !== undefined && !isWireField(raw[f])) return `${f} is invalid`;
  }
  if (!isLabel(raw.label)) return 'label is required';
  if (raw.placeholder !== undefined && !isLabel(raw.placeholder)) {
    return 'placeholder is invalid';
  }
  if (raw.transform !== undefined && !TEXT_TRANSFORMS.includes(raw.transform)) {
    return 'transform must be one of none/uppercase/lowercase/trim';
  }
  return null;
}

const TEXT_COPIED_FIELDS = [
  ...TEXT_WIRE_FIELDS,
  'placeholder',
  'transform',
  'maxLength',
  'monospace',
] as const;

export function parseText(
  raw: unknown,
  key: string,
  errors: string[],
): TextCapability | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": capabilities.text must be an object — skipped`);
    return null;
  }
  const problem = textScalarProblem(raw);
  if (problem) {
    errors.push(`slot "${key}": capabilities.text.${problem} — skipped`);
    return null;
  }
  const pattern = parsePattern(raw.pattern, key, errors);
  if (pattern === null) return null;
  if (raw.maxLength !== undefined && !isValidMaxLength(raw.maxLength)) {
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

  const out: TextCapability = { label: raw.label as string };
  const loose = out as unknown as Record<string, unknown>;
  for (const f of TEXT_COPIED_FIELDS) {
    if (raw[f] !== undefined) loose[f] = raw[f];
  }
  if (pattern !== undefined) out.pattern = pattern;
  if (vocabulary !== undefined) out.vocabulary = vocabulary;
  return out;
}

/* ------------------------------------------------------------------ */
/* provenance                                                          */
/* ------------------------------------------------------------------ */

const PROVENANCE_OPTIONAL = [
  'detectorField',
  'detectorVersionField',
  'chainField',
  'verifierField',
  'verifierVersionField',
  'verifiedAtField',
  'detectedAtField',
] as const;

export function parseProvenance(
  raw: unknown,
  key: string,
  errors: string[],
): ProvenanceCapability | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": capabilities.provenance must be an object — skipped`);
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
  const out: ProvenanceCapability = { showChainOnCard: raw.showChainOnCard };
  for (const f of PROVENANCE_OPTIONAL) {
    if (raw[f] !== undefined) out[f] = raw[f] as string;
  }
  return out;
}
