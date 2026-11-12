import { redirect } from '@sveltejs/kit';
import type { PageLoad } from './$types';

// `/quality` alone (the breadcrumb target) lands on its first tab.
export const load: PageLoad = ({ params, url }) => {
  redirect(307, `/p/${encodeURIComponent(params.project)}/quality/audit${url.search}`);
};
