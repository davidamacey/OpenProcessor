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
const PROJECT_SECTIONS = [
  'dashboard',
  'ingest',
  'datasets',
  'workspace',
  'images',
  'classes',
  'export',
  'quality',
  'models',
  'train',
  'bakeoff',
  'settings',
] as const;
export type ProjectSection = (typeof PROJECT_SECTIONS)[number];

/** The header's primary navigation, in order: one source for the nav links and the breadcrumb. */
export const PRIMARY_NAV: ReadonlyArray<{ section: ProjectSection; label: string }> = [
  { section: 'dashboard', label: 'Dashboard' },
  { section: 'ingest', label: 'Ingest' },
  { section: 'workspace', label: 'Workspace' },
  { section: 'classes', label: 'Classes' },
  { section: 'export', label: 'Export' },
  { section: 'models', label: 'Models' },
  { section: 'train', label: 'Train' },
  { section: 'bakeoff', label: 'Bake-off' },
  { section: 'quality', label: 'Quality' },
  { section: 'settings', label: 'Settings' },
];

/** The nav label of `section`; the section id for one the header does not link. */
export function sectionLabel(section: ProjectSection): string {
  return PRIMARY_NAV.find((n) => n.section === section)?.label ?? section;
}

/** A path inside a project: a known section, plus any sub-path, query
 *  or hash (`/workspace?view=review&queue=to_label`, `/workspace?view=clusters`, `/settings#keyboard`). */
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

/** The Workspace Clusters results for the crops nearest to `cropId` ("More like this"). */
export function similarHref(cropId: string, slug?: string): string {
  return projectHref(
    `/workspace?view=clusters&similar_to=${encodeURIComponent(cropId)}`,
    slug,
  );
}

/** Query params that name something inside ONE project (a crop, a
 *  class id, the Workspace's focused item, open cluster or similarity anchor) and so never
 *  carry over to another project. */
const PROJECT_LOCAL_PARAMS = ['crop_id', 'class', 'focus', 'scope', 'similar_to'];

/**
 * Where the switcher goes: the same section under `toSlug`. Anything
 * that names an object inside the old project is dropped — the
 * `crop_id`/`class`/`focus`/`scope`/`similar_to` query params —
 * because ids don't carry across projects. Every other query param
 * (e.g. `view`, `queue`) is kept. Outside `/p/...` it lands on the default
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
 * The redirect for `/` and the legacy bare paths (`/classes?x=1`,
 * `/export`, ...): the same path under `/p/<defaultSlug>`, query
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
