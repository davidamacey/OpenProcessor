import { redirect } from '@sveltejs/kit';

// The legacy `/` page was merged into `/dashboard` (2026-09) — nothing was
// dropped, see the comment at the top of src/routes/dashboard/+page.svelte
// for exactly where each piece landed. This redirect keeps old bookmarks
// and the top-bar logo (which points at /dashboard directly) consistent.
export const load = () => {
  throw redirect(307, '/dashboard');
};
