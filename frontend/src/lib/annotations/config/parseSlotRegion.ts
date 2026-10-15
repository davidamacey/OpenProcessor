/** `capabilities.subBox` parser: thumbnail, ring, editor and the box fields. */

import type { SubBoxCapability, SubBoxRing } from '../types';
import { PATH_PLACEHOLDERS, RING_CLASS_ALLOWLIST, RING_PRESETS } from './allowLists';
import { validatePathTemplate, renderPathTemplate } from './templatePath';
import { hasForbiddenKey, isPlainObject, isWireField } from './parseCommon';

/* ------------------------------------------------------------------ */
/* subBox                                                              */
/* ------------------------------------------------------------------ */

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
    path: (cropId: string, boxId: string, size: number) =>
      renderPathTemplate(template, { cropId, boxId, size }),
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

const SUBBOX_OPTIONAL_FIELDS = [
  'frameField',
  'scoreField',
  'visibleField',
  'countField',
  'rejectedCountField',
  'maxScoreField',
  'setCompleteField',
  'revisionField',
] as const;

export function parseSubBox(
  raw: unknown,
  key: string,
  errors: string[],
): SubBoxCapability | null | undefined {
  if (raw === undefined) return undefined;
  if (!isPlainObject(raw) || hasForbiddenKey(raw)) {
    errors.push(`slot "${key}": capabilities.subBox must be an object — skipped`);
    return null;
  }
  const hasList = raw.listField !== undefined;
  if (hasList === (raw.bboxField !== undefined)) {
    errors.push(
      `slot "${key}": capabilities.subBox needs exactly one of listField (multi-box) or bboxField (read-only single box) — skipped`,
    );
    return null;
  }
  if (hasList && !isWireField(raw.listField)) {
    errors.push(`slot "${key}": capabilities.subBox.listField is invalid — skipped`);
    return null;
  }
  if (!hasList) {
    if (!isWireField(raw.bboxField)) {
      errors.push(`slot "${key}": capabilities.subBox.bboxField is invalid — skipped`);
      return null;
    }
    if (raw.storedFrame !== 'source' && raw.storedFrame !== 'parent') {
      errors.push(
        `slot "${key}": capabilities.subBox.storedFrame must be 'source' or 'parent' — skipped`,
      );
      return null;
    }
  }
  for (const f of SUBBOX_OPTIONAL_FIELDS) {
    if (raw[f] !== undefined && !isWireField(raw[f])) {
      errors.push(`slot "${key}": capabilities.subBox.${f} is invalid — skipped`);
      return null;
    }
  }
  const thumbnail = parseThumbnail(raw.thumbnail, key, errors);
  if (thumbnail === null) return null;
  const ring = parseRing(raw.ring, key, errors);
  if (ring === null) return null;
  const editor = parseEditor(raw.editor, key, errors);
  if (editor === null) return null;

  const out: SubBoxCapability = { ring, editor };
  if (hasList) {
    out.listField = raw.listField as string;
  } else {
    out.bboxField = raw.bboxField as string;
    out.storedFrame = raw.storedFrame as SubBoxCapability['storedFrame'];
  }
  for (const f of SUBBOX_OPTIONAL_FIELDS) {
    if (raw[f] !== undefined) out[f] = raw[f] as string;
  }
  if (thumbnail !== undefined) out.thumbnail = thumbnail;
  return out;
}
