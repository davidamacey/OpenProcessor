import type { BoxTone } from '$lib/types';

const TONE_CLASS: Record<BoxTone, string> = {
  validated: 'border-success',
  machine: 'border-warn',
  unlabeled: 'border-fg-3',
  holdout: 'border-holdout',
  rejected: 'border-danger',
};

/** The border colour of a served box tone: the one tone-to-class map. */
export function boxToneClass(tone: BoxTone): string {
  return TONE_CLASS[tone];
}
