/**
 * `license_plate` was briefly hidden from class-assignment surfaces
 * (2026-09-12) on the theory that it's a sub-bbox axis (the plate
 * detection/OCR workflow) wearing a vehicle-class-registry id, not a real
 * one-of-N vehicle body-type class, and that its huge cluster-size risked
 * accidental mis-assignment. Reverted the same day: David wants it fully
 * assignable — plates get relabeled/added to over time, and the goal is a
 * future unified vehicle+plate single-pass model, so keeping it a normal,
 * reachable class serves that better than hiding it. `isPickerHiddenClass`
 * is kept as a named, always-false predicate (rather than deleting the
 * concept outright) so every call site still routes through one chokepoint
 * — flipping visibility for a specific class in the future is a one-line
 * change here, not a re-audit of every picker/sidebar/hotkey call site.
 *
 * `isAssignableClass` SUBSUMES the `!c.deprecated` filter every calling
 * surface already applied — callers should use this one, not stack both
 * (a deprecated class was already excluded; adding
 * `!c.deprecated && isAssignableClass(c)` is redundant and just makes the
 * predicate harder to grep for later).
 */

/** Currently hides nothing — see the module header for why this stays a
 *  named predicate instead of being deleted. */
export function isPickerHiddenClass(_name: string): boolean {
  return false;
}

/**
 * True if `cls` may be offered as a class-ASSIGNMENT target (picker row,
 * quick-assign chip, drag-drop zone, confirm-to dropdown option, hotkey
 * binding). False for deprecated classes only today.
 *
 * Never filter `classesStore.classes` itself through this — the store must
 * stay globally unfiltered (the pinned plate-gallery card and its
 * click-routing resolve class 80 directly out of it).
 */
export function isAssignableClass(cls: { name: string; deprecated?: boolean }): boolean {
  if (cls.deprecated) return false;
  if (isPickerHiddenClass(cls.name)) return false;
  return true;
}
