import { regionRuleWarnings } from '$lib/annotations/registeredSlots';
import { loadKeymap } from '$stores/keymap.svelte';
import { loadRegionProfile } from '$stores/regionProfile.svelte';
import { projectsStore, type ProjectResolution } from '$stores/projects.svelte';
import type { LayoutLoad } from './$types';

let ruleWarningsLogged = false;

/**
 * Resolves `/p/<slug>` against the served project list and makes it the
 * active project (the URL path is the only store of the active project).
 * `projectsStore.select()` moves `scoped()` to the project's served
 * `prefix` and — on a change of project — runs every project-change
 * reset hook (undo stack, caches, vocabularies). An unknown slug, or one
 * the server marks not `selectable`, renders the "not available" page
 * instead, and no scoped call fires.
 */
export const load: LayoutLoad = async ({
  params,
  parent,
}): Promise<{ resolution: ProjectResolution | null }> => {
  const { projectsError } = await parent();
  if (projectsError) return { resolution: null };

  const resolution = await projectsStore.resolve(params.project);
  if (resolution.kind !== 'ok') return { resolution };

  projectsStore.select(resolution.project);

  // The served region profile (`{prefix}/health.region_profile`) and the
  // served keymap are per project. This is after the prefix moved and
  // before first render, which is what the slot registry's live
  // bindings need (see annotations/registeredSlots.ts). Both never
  // throw and are bounded at 2 s per try.
  await Promise.all([loadRegionProfile(), loadKeymap()]);
  if (!ruleWarningsLogged) {
    ruleWarningsLogged = true;
    for (const w of regionRuleWarnings) console.warn(`[annotation-profiles] ${w}`);
  }
  return { resolution };
};
