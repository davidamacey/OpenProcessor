import { redirect } from '@sveltejs/kit';
import type { PageLoad } from './$types';

// The `datasets` section root (the shell's breadcrumb links it) lands on
// the imports list.
export const load: PageLoad = async ({ parent, params }) => {
  const { resolution } = await parent();
  if (resolution?.kind !== 'ok') return;
  redirect(307, `/p/${encodeURIComponent(params.project)}/datasets/imports`);
};
