import { apiBase } from '$lib/api';
import { loadDeploymentProfiles } from '$lib/annotations/deploymentProfiles';

// SPA — no SSR, no prerender. The labeler is a private tool that depends on
// the locally-running openprocessor; static-rendered routes would not have a
// useful base URL anyway.
export const ssr = false;
export const prerender = false;
export const trailingSlash = 'never';

export const load = async () => {
  // Tier-2 deployment annotation profiles. This is the one point in the
  // lifecycle that is after module evaluation and before first render,
  // which is exactly what the slot registry's live bindings need (see
  // annotations/registeredSlots.ts). Never throws, bounded at 2 s, and
  // a no-op on a deployment with no profile file — which is every
  // deployment that has not opted in.
  await loadDeploymentProfiles();
  return { apiBase };
};

export type LayoutLoadData = Awaited<ReturnType<typeof load>>;
