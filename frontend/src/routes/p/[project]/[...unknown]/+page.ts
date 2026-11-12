import { error } from '@sveltejs/kit';
import type { PageLoad } from './$types';

// A path under a project that is no page: a 404 raised inside the project layout, so the
// error page renders with the top bar (`../+error.svelte`) instead of on a bare screen.
export const load: PageLoad = () => {
  error(404, 'Not found');
};
