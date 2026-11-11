import type { BoxTone } from '$lib/types';

const TONE_CLASS: Record<BoxTone, string> = {
  validated: 'border-emerald-400',
  machine: 'border-amber-400',
  unlabeled: 'border-zinc-500',
  holdout: 'border-violet-400',
  rejected: 'border-red-400',
};

/** The border colour of a served box tone: the one tone-to-class map. */
export function boxToneClass(tone: BoxTone): string {
  return TONE_CLASS[tone];
}
