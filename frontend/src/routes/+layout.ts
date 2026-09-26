import { apiBase } from '$lib/api';
import { loadDeploymentProfiles } from '$lib/annotations/deploymentProfiles';
import { regionRuleWarnings } from '$lib/annotations/registeredSlots';
import { loadKeymap } from '$stores/keymap.svelte';
import { loadRegionProfile } from '$stores/regionProfile.svelte';

// SPA — no SSR, no prerender. Every page depends on the curation API at
// runtime; static-rendered routes would not have a useful base URL anyway.
export const ssr = false;
export const prerender = false;
export const trailingSlash = 'never';

let ruleWarningsLogged = false;

export const load = async () => {
  // The served region profile (`/health.region_profile`) and tier-2
  // deployment annotation profiles. This is the one point in the
  // lifecycle that is after module evaluation and before first render,
  // which is exactly what the slot registry's live bindings need (see
  // annotations/registeredSlots.ts). Both never throw and are bounded at
  // 2 s; the registry composes them in either completion order.
  await Promise.all([loadRegionProfile(), loadDeploymentProfiles(), loadKeymap()]);
  if (!ruleWarningsLogged) {
    ruleWarningsLogged = true;
    for (const w of regionRuleWarnings) console.warn(`[annotation-profiles] ${w}`);
  }
  return { apiBase };
};

export type LayoutLoadData = Awaited<ReturnType<typeof load>>;
