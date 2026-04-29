import { apiBase } from '$lib/api';

// SPA — no SSR, no prerender. The labeler is a private tool that depends on
// the locally-running openprocessor; static-rendered routes would not have a
// useful base URL anyway.
export const ssr = false;
export const prerender = false;
export const trailingSlash = 'never';

export const load = () => {
  return { apiBase };
};

export type LayoutLoadData = ReturnType<typeof load>;
