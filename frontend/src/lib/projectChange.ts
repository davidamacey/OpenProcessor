/**
 * The project-change reset registry
 * (`docs/design/any-domain-rev3-and-projects-contract-review-2026-09-26.md`
 * §7.1). Every client-side store or cache that holds one project's data
 * registers a hook here from its OWN module; `projectsStore.select()`
 * runs them all when the active project changes, so nothing from the
 * previous project renders in the next one. Crop ids are content-derived
 * (the same image has the same `crop_id` in every project), so an
 * un-reset cache would silently serve the wrong project's data.
 *
 * Deliberately dependency-free: stores import this, never the projects
 * store, so there is no import cycle between the two.
 */

const resetHooks = new Set<() => void>();

export function onProjectChange(hook: () => void): () => void {
  resetHooks.add(hook);
  return () => resetHooks.delete(hook);
}

/** Runs every registered reset hook. Called by `projectsStore.select()`
 *  after the scoped prefix has moved to the new project. */
export function resetForProjectChange(): void {
  for (const hook of resetHooks) hook();
}
