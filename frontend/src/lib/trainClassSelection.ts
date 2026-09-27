/**
 * V-4 (fresh-start coordinator review 2026-09-25): `/train`'s "Classes to
 * train" defaulted to every class, including one with 0 crops, so
 * preflight blocked on class balance / holdout / split coverage.
 *
 * "Enough data" is the server's own call: `GET {API_PREFIX}/classes`
 * serves `trainable_gap` per class — the shortfall of its trainable count
 * (validated minus test holdout minus excluded) against the served
 * per-class hard minimum, floored at 0. A class is lacking when that
 * served gap is > 0. No threshold is computed or invented here.
 */
import type { RegistryClass } from '$lib/types';

function candidates(classes: RegistryClass[]): RegistryClass[] {
  // Region-kind classes are sub-box slots, not item classes a detector
  // run trains on (served `kind`).
  return classes.filter((c) => !c.deprecated && c.kind !== 'region');
}

/** Non-deprecated item classes the server says are short of data. */
export function classesLackingData(classes: RegistryClass[]): RegistryClass[] {
  return candidates(classes).filter((c) => c.trainable_gap > 0);
}

/**
 * The default selection: `null` ("all classes") when no class is short of
 * data, else the explicit list of item classes that aren't.
 */
export function defaultTrainSelection(classes: RegistryClass[]): number[] | null {
  const lacking = new Set(classesLackingData(classes).map((c) => c.id));
  if (lacking.size === 0) return null;
  return candidates(classes)
    .filter((c) => !lacking.has(c.id))
    .map((c) => c.id)
    .sort((a, b) => a - b);
}

/** Drop every class the server says is short of data from `selected`
 *  (`null` = all non-deprecated classes). */
export function excludeLackingData(
  selected: number[] | null,
  classes: RegistryClass[],
): number[] {
  const lacking = new Set(classesLackingData(classes).map((c) => c.id));
  const base =
    selected === null
      ? classes.filter((c) => !c.deprecated).map((c) => c.id)
      : [...selected];
  return base.filter((id) => !lacking.has(id)).sort((a, b) => a - b);
}

/** Selected classes (`null` = all) the server says are short of data. */
export function selectedLackingData(
  selected: number[] | null,
  classes: RegistryClass[],
): RegistryClass[] {
  const lacking = classesLackingData(classes);
  if (selected === null) return lacking;
  const sel = new Set(selected);
  return lacking.filter((c) => sel.has(c.id));
}
