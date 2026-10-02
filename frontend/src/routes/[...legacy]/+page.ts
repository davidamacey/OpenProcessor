import { error, redirect } from '@sveltejs/kit';
import { legacyRedirectTarget } from '$lib/projectPaths';
import { projectsStore } from '$stores/projects.svelte';
import type { PageLoad } from './$types';

// The bare pre-projects paths (`/review?tab=x`, `/clusters/12`, ...) are
// URL aliases for bookmarks: they redirect to the same path under the
// served default project, query string kept. Anything that isn't a
// project section is a 404. `/projects` and `/p/...` are more specific
// routes, so they never reach here.
export const load: PageLoad = async ({ parent, url }) => {
  const { projectsError } = await parent();
  if (projectsError) return;
  const target = projectsStore.defaultProject;
  const to = target ? legacyRedirectTarget(url.pathname, url.search, target.slug) : null;
  if (to) redirect(307, to);
  if (!target && legacyRedirectTarget(url.pathname, url.search, 'x'))
    redirect(307, '/projects');
  error(404, 'Not found');
};
