import type { BakeoffFailure } from './api';

/** "stage · dataset · model" for whichever of the three a failure names;
 *  `job` when it names none. */
export function bakeoffFailureWhere(f: BakeoffFailure): string {
  return [f.stage, f.dataset, f.model].filter(Boolean).join(' · ') || 'job';
}
