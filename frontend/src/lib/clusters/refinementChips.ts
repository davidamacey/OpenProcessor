/**
 * The refinements in force on the cluster list, one chip each: the card filter, the subject
 * scope, the clarity floor and the modes. Labels come from the served flag vocabulary and the
 * served validation words; each chip knows how to clear itself. The Refine button's count is
 * the number of chips.
 */
import { VALIDATION_TEXT } from '$lib/clusters/clusterCardText';
import type { ClusterFlag, ClusterListFilter } from '$lib/types_clusterList';

export interface RefinementChip {
  id: string;
  label: string;
  clear: () => void;
}

export interface RefinementState {
  filter: ClusterListFilter;
  flagVocabulary: readonly ClusterFlag[];
  setFilter: (next: ClusterListFilter) => void;
  subjectScope: 0 | 1 | 2;
  setSubjectScope: (v: 0 | 1 | 2) => void;
  /** The committed clarity floor, or null for none. */
  minBlurRatio: number | null;
  clearClarity: () => void;
}

const SUBJECT = ['', 'Largest', 'Largest and second'] as const;

function sizeLabel(min: number | null, max: number | null): string {
  if (min != null && max != null) return `Size ${min} to ${max}`;
  return min != null ? `Size at least ${min}` : `Size at most ${max}`;
}

export function refinementChips(s: RefinementState): RefinementChip[] {
  const f = s.filter;
  const chips: RefinementChip[] = [];
  for (const name of f.dominant_class)
    chips.push({
      id: `dominant:${name}`,
      label: `Class: ${name}`,
      clear: () =>
        s.setFilter({ ...f, dominant_class: f.dominant_class.filter((x) => x !== name) }),
    });
  for (const name of f.detector_class)
    chips.push({
      id: `detector:${name}`,
      label: `Detector class: ${name}`,
      clear: () =>
        s.setFilter({ ...f, detector_class: f.detector_class.filter((x) => x !== name) }),
    });
  if (f.min_size != null || f.max_size != null)
    chips.push({
      id: 'size',
      label: sizeLabel(f.min_size, f.max_size),
      clear: () => s.setFilter({ ...f, min_size: null, max_size: null }),
    });
  if (f.validation_state)
    chips.push({
      id: 'validation',
      label: `Validation: ${VALIDATION_TEXT[f.validation_state]}`,
      clear: () => s.setFilter({ ...f, validation_state: null }),
    });
  if (f.needs_review)
    chips.push({
      id: 'needs_review',
      label: 'Needs review',
      clear: () => s.setFilter({ ...f, needs_review: false }),
    });
  for (const id of f.flag)
    chips.push({
      id: `flag:${id}`,
      label: s.flagVocabulary.find((v) => v.id === id)?.label ?? id,
      clear: () => s.setFilter({ ...f, flag: f.flag.filter((x) => x !== id) }),
    });
  if (s.subjectScope !== 0)
    chips.push({
      id: 'subject',
      label: `Subject: ${SUBJECT[s.subjectScope]}`,
      clear: () => s.setSubjectScope(0),
    });
  if (s.minBlurRatio != null)
    chips.push({ id: 'clarity', label: 'Minimum clarity set', clear: s.clearClarity });
  return chips;
}
