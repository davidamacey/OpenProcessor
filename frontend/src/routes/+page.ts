import { redirect } from '@sveltejs/kit';
import { legacyRedirectTarget } from '$lib/projectPaths';
import { projectsStore } from '$stores/projects.svelte';
import type { PageLoad } from './$types';

// `/` is not a page: it redirects to the served default project's
// dashboard (`/p/<default_slug>/dashboard`), query string kept. With no
// selectable project at all, the project list is the only useful place.
export const load: PageLoad = async ({ parent, url }) => {
  const { projectsError } = await parent();
  if (projectsError) return;
  const target = projectsStore.defaultProject;
  if (!target) redirect(307, '/projects');
  redirect(307, legacyRedirectTarget('/', url.search, target.slug)!);
};
