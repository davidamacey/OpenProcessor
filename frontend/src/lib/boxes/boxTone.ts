import type { BoxTone } from '$lib/types';

const TONE_CLASS: Record<BoxTone, string> = {
  validated: 'border-success',
  machine: 'border-warn',
  unlabeled: 'border-fg-3',
  holdout: 'border-holdout',
  rejected: 'border-danger',
};

const DETECTOR_CLASS = 'border-tone-detector';

/** The border colour of a served box tone: the one tone-to-class map. A box that is only the
 *  detector's proposal (served `label_kind: detector`, not yet decided) takes the detector tone. */
export function boxToneClass(tone: BoxTone, labelKind?: string): string {
  if (labelKind === 'detector' && (tone === 'unlabeled' || tone === 'machine'))
    return DETECTOR_CLASS;
  return TONE_CLASS[tone];
}
