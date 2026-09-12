/**
 * `license_plate` (class_id 80 as of 2026-09, verify via `GET /curation/classes` —
 * registry ids can drift) is a structural singleton: it's a sub-bbox axis
 * (the plate detection/OCR workflow, see PlateCard.svelte/DetectorChip.svelte)
 * wearing a vehicle-class-registry id, not a real one-of-N vehicle body-type
 * class. It sorts near the top of every class picker by cluster-size
 * (55,937 crops carry plate-related data as of writing), which risks
 * accidental mis-assignment to a class that isn't a real labeling target.
 *
 * This module is the single chokepoint every class-ASSIGNMENT surface
 * (fuzzy-search picker, quick-assign row, sidebar drag target, confirm-to
 * dropdowns, embedding-plot assign select, hotkey editor/binding) routes
 * through to hide it. It is deliberately NOT hidden from non-assignment
 * surfaces — stats, the review queue's class FILTER dropdown, `/export` and
 * `/train`'s class-subset picker, and `/classes` admin CRUD all keep using
 * the raw, unfiltered `classesStore.classes`/`byId`/`byName`.
 *
 * `isAssignableClass` SUBSUMES the `!c.deprecated` filter every one of those
 * surfaces already applied — callers should replace that filter with this
 * one, not stack both (a deprecated class was already excluded; adding
 * `!c.deprecated && isAssignableClass(c)` is redundant and just makes the
 * predicate harder to grep for later).
 *
 * A `hidden_from_picker: bool` registry field was considered and rejected in
 * favor of this hardcoded name-check: `license_plate` is a one-off (six
 * other call sites in this codebase already hardcode its name for other
 * reasons — sidebar routing to the plate gallery, the synthetic gallery-card
 * builder, cluster-click routing), so a predicate matches this codebase's
 * existing idiom instead of introducing a second parallel mechanism.
 *
 * Un-hide upgrade path: if a future unified vehicle+plate model needs
 * `license_plate` assignable again, swap this function's body for
 * `cls.hidden_from_picker ?? isLicensePlateName(cls.name)` once that backend
 * field exists — every caller already routes through here, so it's a
 * one-line change.
 */

/** Exact match only (case-insensitive) — must NOT catch a hypothetical
 *  `license_plate_holder` or similar near-miss name via substring match. */
function isLicensePlateName(name: string): boolean {
  return name.toLowerCase() === 'license_plate';
}

/** Bare predicate for use in non-OpClass contexts (e.g. a keydown guard that
 *  only has the class name in hand, not the full registry row). */
export function isPickerHiddenClass(name: string): boolean {
  return isLicensePlateName(name);
}

/**
 * True if `cls` may be offered as a class-ASSIGNMENT target (picker row,
 * quick-assign chip, drag-drop zone, confirm-to dropdown option, hotkey
 * binding). False for deprecated classes and for `license_plate`.
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
