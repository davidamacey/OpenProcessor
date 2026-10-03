/**
 * The backend's class-name rule: names compare normalized (case, spaces and
 * hyphens fold to `_`); an ACTIVE class wins over a deprecated one with the
 * same name; a deprecated-only match is never assigned to. By-name list
 * filters match the name stored on each item, so exact class selection
 * (a cluster card click) uses `class_id`, not this lookup.
 */
export function normalizeClassName(name: string): string {
  return name.trim().toLowerCase().replace(/[\s-]/g, '_');
}

/** The ACTIVE class whose normalized name equals `name`'s; `null` when none
 *  (a deprecated-only match is not a match). */
export function findActiveClassByName<T extends { name: string; deprecated?: boolean }>(
  classes: readonly T[],
  name: string,
): T | null {
  const key = normalizeClassName(name);
  return classes.find((c) => !c.deprecated && normalizeClassName(c.name) === key) ?? null;
}
