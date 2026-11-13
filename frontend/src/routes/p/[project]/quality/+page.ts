import { redirect } from '@sveltejs/kit';
import type { PageLoad } from './$types';

// `/quality` alone (the breadcrumb target) lands on Ready to train.
export const load: PageLoad = ({ params, url }) => {
  redirect(307, `/p/${encodeURIComponent(params.project)}/quality/ready${url.search}`);
};
