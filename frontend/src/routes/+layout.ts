import { apiBase } from '$lib/api';
import { loadDeploymentProfiles } from '$lib/annotations/deploymentProfiles';
import { projectsStore } from '$stores/projects.svelte';

// SPA — no SSR, no prerender. Every page depends on the curation API at
// runtime; static-rendered routes would not have a useful base URL anyway.
export const ssr = false;
export const prerender = false;
export const trailingSlash = 'never';

export const load = async () => {
  // Global reads only. The served project list comes first: every
  // `/p/[project]` route resolves its slug against it, and `/` plus the
  // legacy bare paths redirect to its default project. A failed load
  // leaves `projectsStore.error` set; the root layout renders a
  // blocking error state instead of the app. Project-scoped boot reads
  // (region profile, keymap) live in `p/[project]/+layout.ts`.
  await projectsStore.load();
  if (projectsStore.error) {
    return { apiBase, projectsError: projectsStore.error };
  }
  // Tier-2 deployment annotation profiles: a static file next to
  // index.html, the same for every project. Never throws; bounded.
  await loadDeploymentProfiles();
  return { apiBase, projectsError: null };
};

export type LayoutLoadData = Awaited<ReturnType<typeof load>>;
