/**
 * Every in-app URL is built here (owner decision: the active project
 * lives in the URL path ONLY, `/p/<slug>/...`). Call sites never
 * hand-assemble `/p/...`: links and `goto()`s go through `projectHref()`
 * (wrapped in SvelteKit's `resolve()` at the call site, which the
 * `svelte/no-navigation-without-resolve` lint rule requires), the
 * switcher through `switchProjectHref()`, and the `/` + legacy bare-path
 * redirects through `legacyRedirectTarget()`.
 */

import { ProjectNotSelectedError } from '$lib/api';
import { projectsStore } from '$stores/projects.svelte';

/** The page sections that live under `/p/[project]/`. */
export const PROJECT_SECTIONS = [
  'dashboard',
  'ingest',
  'datasets',
  'clusters',
  'review',
  'classes',
  'export',
  'models',
  'train',
  'bakeoff',
  'settings',
] as const;
export type ProjectSection = (typeof PROJECT_SECTIONS)[number];

/** A path inside a project: a known section, plus any sub-path, query
 *  or hash (`/review?tab=all`, `/clusters/12`, `/settings#keyboard`). */
export type ProjectSubPath = `/${ProjectSection}${string}`;

/** The section a project lands on when none is named. */
export const DEFAULT_SECTION: ProjectSection = 'dashboard';

function isSection(s: string): s is ProjectSection {
  return (PROJECT_SECTIONS as readonly string[]).includes(s);
}

/**
 * `/p/<slug><path>` for the active project (or an explicit `slug`).
 * Throws `ProjectNotSelectedError` when there is no active project —
 * only code rendered under `/p/[project]` builds project links.
 */
export function projectHref<P extends ProjectSubPath>(path: P, slug?: string): string {
  const s = slug ?? projectsStore.current?.slug;
  if (!s) throw new ProjectNotSelectedError();
  return `/p/${encodeURIComponent(s)}${path}`;
}

/** Query params that name something inside ONE project (a crop, a
 *  class id) and so never carry over to another project. */
const PROJECT_LOCAL_PARAMS = ['crop_id', 'class'];

/**
 * Where the switcher goes: the same section under `toSlug`. Anything
 * that names an object inside the old project is dropped — the
 * `/clusters/<id>` id segment and the `crop_id`/`class` query params —
 * because ids don't carry across projects. Every other query param
 * (e.g. `tab`) is kept. Outside `/p/...` it lands on the default
 * section.
 */
export function switchProjectHref(
  url: Pick<URL, 'pathname' | 'search'>,
  toSlug: string,
): string {
  const parts = url.pathname.split('/').filter(Boolean);
  const section = parts[0] === 'p' && parts[2] && isSection(parts[2]) ? parts[2] : null;
  if (!section) return `/p/${encodeURIComponent(toSlug)}/${DEFAULT_SECTION}`;
  const params = new URLSearchParams(url.search);
  for (const k of PROJECT_LOCAL_PARAMS) params.delete(k);
  const q = params.toString();
  return `/p/${encodeURIComponent(toSlug)}/${section}${q ? `?${q}` : ''}`;
}

/**
 * The redirect for `/` and the legacy bare paths (`/review?tab=x`,
 * `/clusters/12`, ...): the same path under `/p/<defaultSlug>`, query
 * string kept. `/` goes to the default section. `null` when the first
 * segment isn't a project section (the caller answers 404).
 */
export function legacyRedirectTarget(
  pathname: string,
  search: string,
  defaultSlug: string,
): string | null {
  const parts = pathname.split('/').filter(Boolean);
  const slug = encodeURIComponent(defaultSlug);
  if (parts.length === 0) return `/p/${slug}/${DEFAULT_SECTION}${search}`;
  if (!isSection(parts[0]!)) return null;
  return `/p/${slug}/${parts.join('/')}${search}`;
}

/** The section of a `/p/<slug>/<section>/...` pathname, if any. */
export function sectionOf(pathname: string): ProjectSection | null {
  const parts = pathname.split('/').filter(Boolean);
  return parts[0] === 'p' && parts[2] && isSection(parts[2]) ? parts[2] : null;
}
