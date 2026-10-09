/**
 * Label-confirmation display text (#119). A VLM label is a suggestion until
 * a human validates it, and the detector's own answer is kept beside it so
 * the two can be compared. Nothing here decides anything: the served
 * catalog role and the served detector fields go in, wording comes out.
 */
import type { ClassSourceRole } from './api';
import type { Crop } from './types';

/** The detector's own class next to a VLM label: only for a VLM-sourced
 *  class that carries the detector's answer, else null. */
export function detectorClassText(
  role: ClassSourceRole | null | undefined,
  crop: Pick<Crop, 'detector_class_name' | 'detector_confidence'>,
): string | null {
  if (role !== 'vlm' || !crop.detector_class_name) return null;
  const conf = crop.detector_confidence;
  return conf == null
    ? crop.detector_class_name
    : `${crop.detector_class_name} ${(conf * 100).toFixed(0)}%`;
}

/** A served ratio in [0, 1] as a percent; "—" when absent. */
export function percentText(v: number | null | undefined, digits = 0): string {
  return v == null ? '—' : `${(v * 100).toFixed(digits)}%`;
}
