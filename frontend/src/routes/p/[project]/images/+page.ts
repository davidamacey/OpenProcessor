import { redirect } from '@sveltejs/kit';
import { projectHref } from '$lib/projectPaths';
import type { PageLoad } from './$types';

// An image is opened by id; the bare section lands on the Workspace.
export const load: PageLoad = async ({ parent, params, url }) => {
  const { resolution } = await parent();
  if (resolution?.kind !== 'ok') return;
  redirect(307, projectHref(`/workspace${url.search}`, params.project));
};
