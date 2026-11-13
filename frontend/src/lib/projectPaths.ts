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

/** The nav groups, in header order. A group with one section is a plain link; a group with
 *  several is a menu. */
const NAV_GROUP_LABELS = {
  home: 'Dashboard',
  data: 'Data',
  work: 'Workspace',
  quality: 'Data quality',
  model: 'Model',
  settings: 'Settings',
} as const;
export type NavGroupId = keyof typeof NAV_GROUP_LABELS;

/**
 * The one navigation table: every page section under `/p/[project]/`, in the order the header
 * shows them, with its human label and its group. The header nav, the mobile drawer, the
 * breadcrumb and the project switcher all read this; nothing else spells a section label.
 * `hidden` sections have a page but no nav entry (reached from another page).
 */
const SECTION_TABLE = [
  { section: 'dashboard', label: 'Dashboard', group: 'home' },
  { section: 'ingest', label: 'Ingest', group: 'data' },
  { section: 'classes', label: 'Classes', group: 'data' },
  { section: 'images', label: 'Images', group: 'data', hidden: true },
  { section: 'workspace', label: 'Workspace', group: 'work' },
  { section: 'quality', label: 'Data quality', group: 'quality' },
  { section: 'export', label: 'Export', group: 'model' },
  { section: 'train', label: 'Train', group: 'model' },
  { section: 'bakeoff', label: 'Bake-off', group: 'model' },
  { section: 'models', label: 'Models', group: 'model' },
  { section: 'settings', label: 'Settings', group: 'settings' },
] as const;

const PROJECT_SECTIONS = SECTION_TABLE.map((r) => r.section);
export type ProjectSection = (typeof SECTION_TABLE)[number]['section'];

/** The header's primary navigation, in order: the sections that have a nav entry. */
export const PRIMARY_NAV: ReadonlyArray<{ section: ProjectSection; label: string }> =
  SECTION_TABLE.filter((r) => !('hidden' in r)).map(({ section, label }) => ({
    section,
    label,
  }));

export interface NavGroup {
  id: NavGroupId;
  label: string;
  sections: ReadonlyArray<{ section: ProjectSection; label: string }>;
}

/** PRIMARY_NAV folded into its groups, for the grouped header. */
export const NAV_GROUPS: ReadonlyArray<NavGroup> = (
  Object.keys(NAV_GROUP_LABELS) as NavGroupId[]
).map((id) => ({
  id,
  label: NAV_GROUP_LABELS[id],
  sections: SECTION_TABLE.filter((r) => r.group === id && !('hidden' in r)).map(
    ({ section, label }) => ({ section, label }),
  ),
}));

/** The human label of `section`. */
export function sectionLabel(section: ProjectSection): string {
  return SECTION_TABLE.find((r) => r.section === section)!.label;
}

/** The nav group a section sits in. */
export function navGroupOf(section: ProjectSection): NavGroupId {
  return SECTION_TABLE.find((r) => r.section === section)!.group;
}

/** Labels of the known sub-pages (the path segment after the section). A segment not listed
 *  here is an object's name or id and is shown as it is. */
const SUBPAGE_LABELS: Partial<Record<ProjectSection, Record<string, string>>> = {
  quality: {
    ready: 'Ready to train',
    audit: 'Accuracy audit',
    history: 'Label history',
  },
  settings: {
    'embedding-spaces': 'Embedding spaces',
    'ingest-policy': 'Ingest policy',
    keyboard: 'Keyboard shortcuts',
    'cluster-profiles': 'Cluster profiles',
    'assist-loop-presets': 'Assist-loop presets',
    'near-dup-profiles': 'Near-duplicate profiles',
    'serving-profiles': 'Serving profiles',
    scores: 'Curation scores',
    models: 'Models',
    'open-vocab': 'Open vocabulary',
    presets: 'Presets',
    'prompt-packs': 'Prompt packs',
    'region-profiles': 'Region profiles',
  },
};

export interface Crumb {
  label: string;
  /** An app path for `resolve()`; absent for the group (no page) and for the current page. */
  href?: string;
}

/** Whether `rest` (the path after the section) is an object page, the only pages that show a
 *  breadcrumb: the import job, the image editor, a settings item editor. A section or a tab
 *  page is marked by the nav and the tab strip. */
function isObjectPage(section: ProjectSection, rest: string[]): boolean {
  if (section === 'ingest') return rest[0] === 'imports' && rest.length === 2;
  if (section === 'images') return rest.length === 1;
  if (section === 'settings') return rest.length >= 2;
  return false;
}

/**
 * The breadcrumb of an object page: the nav group (when it has several sections), the section,
 * then each sub-page and the object. `objectName` is the object's served name (a file name, an
 * import name); without it the path segment is shown. Empty on every other page and outside
 * `/p/<slug>/<section>`.
 */
export function breadcrumbOf(
  pathname: string,
  slug?: string,
  objectName?: string,
): Crumb[] {
  const parts = pathname.split('/').filter(Boolean);
  const section = parts[0] === 'p' && parts[2] && isSection(parts[2]) ? parts[2] : null;
  if (!section) return [];
  const rest = parts.slice(3);
  if (!isObjectPage(section, rest)) return [];
  const group = NAV_GROUPS.find((g) => g.id === navGroupOf(section))!;
  const crumbs: Crumb[] = [];
  if (group.sections.length > 1) crumbs.push({ label: group.label });
  // The import job belongs to the Import tab of Ingest, so its trail goes back to that tab.
  const sectionPath = section === 'ingest' ? '/ingest?tab=import' : `/${section}`;
  crumbs.push({
    label: sectionLabel(section),
    href: projectHref(sectionPath as ProjectSubPath, slug),
  });
  const labels = SUBPAGE_LABELS[section] ?? {};
  const segments = section === 'ingest' ? rest.slice(1) : rest;
  let path = `/${section}${section === 'ingest' ? '/imports' : ''}`;
  segments.forEach((seg, i) => {
    path += `/${seg}`;
    const last = i === segments.length - 1;
    crumbs.push({
      label: last && objectName ? objectName : (labels[seg] ?? decodeURIComponent(seg)),
      ...(last ? {} : { href: projectHref(path as ProjectSubPath, slug) }),
    });
  });
  return crumbs;
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
