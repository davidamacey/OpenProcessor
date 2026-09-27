import { redirect } from '@sveltejs/kit';
import { DEFAULT_SECTION } from '$lib/projectPaths';
import type { PageLoad } from './$types';

// `/p/<slug>` alone lands on the project's default section. The layout's
// load already resolved the slug; an unknown one renders the layout's
// "not available" page instead of this redirect's target.
export const load: PageLoad = async ({ parent, params, url }) => {
  const { resolution } = await parent();
  if (resolution?.kind !== 'ok') return;
  redirect(
    307,
    `/p/${encodeURIComponent(params.project)}/${DEFAULT_SECTION}${url.search}`,
  );
};
