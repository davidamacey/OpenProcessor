/**
 * Compact display text for `CropCard` (visual audit 2026-09-24, K2/K4).
 *
 * At 8 grid columns the card's label row read "Labeled by ... spor... 92%":
 * the served source label ("Labeled by the VLM") ate the width the class
 * name needed. The chip now shows a short code keyed off the served
 * catalog ROLE (`GET /class_sources`), with the full served label in its
 * tooltip — no source is ever recognised by its id string.
 */
import type { ClassSourceRole } from '$lib/api';

const ROLE_SHORT: Record<string, string> = {
  human: 'H',
  vlm: 'VLM',
  model: 'M',
  cluster: 'C',
  proposal: 'P',
  low_conf: 'P',
};

/** Short chip text for a label source; the served label goes in `title`. */
export function sourceShortCode(
  role: ClassSourceRole | null | undefined,
  label: string,
): string {
  const r = role ?? '';
  if (r in ROLE_SHORT) return ROLE_SHORT[r]!;
  if (r.startsWith('vlm')) return 'VLM';
  const fallback = label.trim();
  return fallback ? fallback.slice(0, 2).toUpperCase() : '?';
}

/**
 * Readable text for the served `vlm_class_empty_reason` id. The backend
 * serves no label vocabulary for it yet, so known ids get a sentence and
 * anything else renders humanised (underscores to spaces), never dropped.
 */
export function vlmEmptyReasonText(reason: string): string {
  switch (reason) {
    case 'no_answer':
      return 'VLM gave no answer';
    case 'no_match':
      return 'VLM answer matched no class';
    default:
      return `VLM gave no class (${reason.replace(/_/g, ' ')})`;
  }
}
