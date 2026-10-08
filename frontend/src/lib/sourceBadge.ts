/**
 * The crop card's label-source badge. `label_source` uses the
 * `class_source` vocabulary: color comes from the backend's catalog role
 * (`GET /class_sources`) and text from its label. An id outside the
 * catalog renders verbatim in a neutral tone — the frontend never infers
 * meaning from a source's name.
 */
import type { ClassSourceRole } from './api';

const TONE = {
  human: 'bg-green-500/20 text-green-200 border-green-500/40',
  vlm: 'bg-yellow-500/20 text-yellow-200 border-yellow-500/40',
  cluster: 'bg-purple-500/20 text-purple-200 border-purple-500/40',
  proposal: 'bg-zinc-700/40 text-zinc-300 border-zinc-600',
  model: 'bg-blue-500/20 text-blue-200 border-blue-500/40',
  openVocab: 'bg-sky-500/20 text-sky-200 border-sky-500/40',
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
  if (r === 'human') return { text: 'human', cls: TONE.human, unvalidated: false };
  if (r.startsWith('vlm')) {
    return validated
      ? { text, cls: TONE.vlm, unvalidated: false }
      : { text: `${text} ·`, cls: TONE.vlm, unvalidated: true };
  }
  if (r === 'cluster') return { text, cls: TONE.cluster, unvalidated: false };
  if (r === 'proposal' || r === 'low_conf')
    return { text, cls: TONE.proposal, unvalidated: false };
  if (r === 'model') return { text, cls: TONE.model, unvalidated: false };
  if (r === 'open_vocab') return { text, cls: TONE.openVocab, unvalidated: false };
  return { text, cls: TONE.proposal, unvalidated: false };
}
