/**
 * The crop card's label-source badge. `label_source` uses the
 * `class_source` vocabulary: color comes from the backend's catalog role
 * (`GET /class_sources`) and text from its label. An id outside the
 * catalog renders verbatim in a neutral tone — the frontend never infers
 * meaning from a source's name.
 */
import type { ClassSourceRole } from '$lib/api_classes';

const TONE = {
  human: 'bg-success/20 text-success border-success/40',
  vlm: 'bg-warn/20 text-warn border-warn/40',
  cluster: 'bg-holdout/20 text-holdout border-holdout/40',
  proposal: 'bg-surface-3/40 text-fg-2 border-line-strong',
  model: 'bg-accent/20 text-accent border-accent/40',
  openVocab: 'bg-info/20 text-info border-info/40',
} as const;

/** The confirmation wording (#119): what a class label is worth as ground
 *  truth, keyed off the served catalog role and the validated flag only. */
const CONFIRMATION = {
  human: 'Human-confirmed',
  vlm: 'VLM suggestion',
  cluster: 'Auto-validated',
} as const;

export interface SourceBadge {
  text: string;
  cls: string;
  /** p4 (2026-09-24 interactive pass): whether `text` carries the
   *  "unvalidated" marker — was a bare trailing "?" appended straight
   *  into the badge text, which read as a question ("Labeled by the
   *  VLM?") rather than a validation-state affordance. Callers use this
   *  to put the real explanation in a `title`/tooltip instead. */
  unvalidated: boolean;
}

/** Roles whose badge is just the label in a fixed tone. */
const PLAIN_ROLE_TONE: Record<string, string> = {
  proposal: TONE.proposal,
  low_conf: TONE.proposal,
  model: TONE.model,
  open_vocab: TONE.openVocab,
};

function vlmBadge(role: string, text: string, validated: boolean): SourceBadge {
  // A VLM label that no human validated is a suggestion, never "labeled".
  if (role === 'vlm' && !validated) {
    return { text: CONFIRMATION.vlm, cls: TONE.vlm, unvalidated: true };
  }
  return validated
    ? { text, cls: TONE.vlm, unvalidated: false }
    : { text: `${text} ·`, cls: TONE.vlm, unvalidated: true };
}

export function sourceBadge(
  src: string | null | undefined,
  validated: boolean,
  role: ClassSourceRole | null,
  label: string,
): SourceBadge {
  if (!src || src === 'unknown') {
    return {
      text: validated ? 'auto' : 'unlabeled',
      cls: TONE.proposal,
      unvalidated: false,
    };
  }
  const r = role ?? '';
  const text = label || src;
  if (r === 'human')
    return { text: CONFIRMATION.human, cls: TONE.human, unvalidated: false };
  if (r.startsWith('vlm')) return vlmBadge(r, text, validated);
  if (r === 'cluster')
    return {
      text: validated ? CONFIRMATION.cluster : text,
      cls: TONE.cluster,
      unvalidated: false,
    };
  const cls = PLAIN_ROLE_TONE[r] ?? TONE.proposal;
  return { text, cls, unvalidated: false };
}

/** The wording for a class label's confirmation state, or null when the
 *  role has none (a model, a proposal, an import: shown by its served
 *  label). */
export function confirmationLabel(
  role: ClassSourceRole | null | undefined,
  validated: boolean,
): string | null {
  if (role === 'human') return CONFIRMATION.human;
  if (role === 'vlm' && !validated) return CONFIRMATION.vlm;
  if (role === 'cluster' && validated) return CONFIRMATION.cluster;
  return null;
}
