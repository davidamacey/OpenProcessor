/**
 * The Settings sections, in one static table: the sidebar groups, the in-page scope badge and
 * the sidebar search all read it. It is navigation only (labels, paths, keywords); every value on
 * a page is served by that page's own routes. A section whose feature the backend does not serve
 * (`needs`) is absent from the nav, as its card used to be absent from the old page.
 */
import { PRESET_KINDS, presetPath } from '$lib/presets/presetKinds';
import type { ProjectSubPath } from '$lib/projectPaths';

export type SettingsScope = 'Project' | 'Deployment' | 'Personal';
export type SettingsGroupId = 'project' | 'labelling' | 'deployment' | 'personal';
export type SettingsNeed = 'packs' | 'profiles' | 'vlm' | 'openVocab' | 'keymap';

export interface SettingsSection {
  id: string;
  label: string;
  /** The path under the project, e.g. `/settings/ingest-policy`. */
  path: ProjectSubPath;
  group: SettingsGroupId;
  scope: SettingsScope;
  /** Extra words the sidebar search matches. */
  keywords: string;
  needs?: SettingsNeed;
}

export const SETTINGS_GROUP_LABELS: Record<SettingsGroupId, string> = {
  project: 'Project',
  labelling: 'Labelling AI',
  deployment: 'Deployment',
  personal: 'Personal',
};

const preset = (id: string): SettingsSection => {
  const k = PRESET_KINDS.find((p) => p.id === id)!;
  return {
    id: k.slug,
    label: k.title,
    path: presetPath(k) as ProjectSubPath,
    group: k.scope === 'deployment' ? 'deployment' : 'project',
    scope: k.scope === 'deployment' ? 'Deployment' : 'Project',
    keywords: k.noun,
  };
};

export const SETTINGS_SECTIONS: readonly SettingsSection[] = [
  {
    id: 'general',
    label: 'General',
    path: '/settings',
    group: 'project',
    scope: 'Deployment',
    keywords: 'defaults sort strategy VLM scope budget confirm',
  },
  {
    id: 'ingest-policy',
    label: 'Ingest policy',
    path: '/settings/ingest-policy',
    group: 'project',
    scope: 'Project',
    keywords: 'detections embedding mode detector class',
  },
  {
    id: 'open-vocab',
    label: 'Open vocabulary',
    path: '/settings/open-vocab',
    group: 'project',
    scope: 'Project',
    keywords: 'SAM prompts words sets',
    needs: 'openVocab',
  },
  {
    id: 'embedding-spaces',
    label: 'Embedding spaces',
    path: '/settings/embedding-spaces',
    group: 'project',
    scope: 'Project',
    keywords: 'vectors encoder re-index build',
  },
  preset('cluster_profile'),
  preset('assist_loop_preset'),
  preset('near_dup_profile'),
  {
    id: 'prompt-packs',
    label: 'Prompt packs',
    path: '/settings/prompt-packs',
    group: 'labelling',
    scope: 'Project',
    keywords: 'VLM instructions prompts revision',
    needs: 'packs',
  },
  {
    id: 'region-profiles',
    label: 'Region profiles',
    path: '/settings/region-profiles',
    group: 'labelling',
    scope: 'Project',
    keywords: 'regions boxes models detection',
    needs: 'profiles',
  },
  {
    id: 'models',
    label: 'Models and VLM',
    path: '/settings/models',
    group: 'labelling',
    scope: 'Project',
    keywords: 'endpoints vlm register activate',
    needs: 'vlm',
  },
  {
    id: 'scores',
    label: 'Curation scores',
    path: '/settings/scores',
    group: 'labelling',
    scope: 'Project',
    keywords: 'coverage compute scorers',
  },
  preset('serving_profile'),
  {
    id: 'keyboard',
    label: 'Keyboard shortcuts',
    path: '/settings/keyboard',
    group: 'personal',
    scope: 'Project',
    keywords: 'keys hotkeys keymap bindings',
    needs: 'keymap',
  },
];

/** The section a pathname belongs to (a sub-page such as `models/vlm/x` belongs to `models`). */
export function sectionOfPath(pathname: string): SettingsSection | null {
  const rel = pathname.replace(/^\/p\/[^/]+/, '');
  let best: SettingsSection | null = null;
  for (const s of SETTINGS_SECTIONS) {
    const at = rel === s.path || rel.startsWith(`${s.path}/`);
    if (at && (!best || s.path.length > best.path.length)) best = s;
  }
  return best;
}

/** Sections whose words match every term of `query` (empty query matches all). */
export function searchSections(
  sections: readonly SettingsSection[],
  query: string,
): SettingsSection[] {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  return sections.filter((s) => {
    const hay =
      `${s.label} ${s.keywords} ${s.scope} ${SETTINGS_GROUP_LABELS[s.group]}`.toLowerCase();
    return terms.every((t) => hay.includes(t));
  });
}
