/**
 * The four research preset kinds: the route segment each has in the app, its scope, and its
 * activation axes (`ACTIVATION_AXES` in `src/db/models/config.py`; the API serves no axis list). The
 * server serves the documents; this file only names the pages.
 */
import type { PresetKindId } from '$lib/types_presets';

export interface PresetAxis {
  /** The axis id the activation routes take. */
  id: string;
  label: string;
}

export interface PresetKind {
  id: PresetKindId;
  /** The path segment of the page: `/settings/presets/<slug>`. */
  slug: string;
  scope: 'deployment' | 'project';
  title: string;
  /** "serving profile": the singular, lower case. */
  noun: string;
  blurb: string;
  /** Empty for the deployment-wide serving profile. */
  axes: readonly PresetAxis[];
}

export const PRESET_KINDS: readonly PresetKind[] = [
  {
    id: 'serving_profile',
    slug: 'serving-profiles',
    scope: 'deployment',
    title: 'Serving profiles',
    noun: 'serving profile',
    blurb:
      'Which Triton servers run, which models stay resident on each, how many instances, image preparation and request priorities. One is active for the whole deployment.',
    axes: [],
  },
  {
    id: 'cluster_profile',
    slug: 'cluster-profiles',
    scope: 'project',
    title: 'Cluster profiles',
    noun: 'cluster profile',
    blurb:
      'How clusters are built: method, embedding field, refresh rule, activation and backend. A profile is active per target, items and region boxes.',
    axes: [
      { id: 'clustering_item', label: 'Items' },
      { id: 'clustering_region_box', label: 'Region boxes' },
    ],
  },
  {
    id: 'assist_loop_preset',
    slug: 'assist-loop-presets',
    scope: 'project',
    title: 'Assist-loop presets',
    noun: 'assist-loop preset',
    blurb:
      'The cluster-sampled auto-label loop: gate, recursion, budget, what to do on budget, and concurrency.',
    axes: [{ id: 'assist_loop_preset', label: 'Assist loop' }],
  },
  {
    id: 'near_dup_profile',
    slug: 'near-dup-profiles',
    scope: 'project',
    title: 'Near-duplicate profiles',
    noun: 'near-duplicate profile',
    blurb:
      'The thresholds that decide near-duplicates and leaks for the image and item levels: dedup and leak cosine, largest group, neighbours.',
    axes: [{ id: 'near_dup', label: 'Near-duplicates' }],
  },
];

/** What activating says about its reach, from the kind's scope. */
export function activationReach(kind: PresetKind): string {
  return kind.scope === 'deployment'
    ? 'This is deployment-wide: it applies to every project.'
    : 'This applies to this project only.';
}

export function presetKindBySlug(slug: string | undefined): PresetKind | null {
  return PRESET_KINDS.find((k) => k.slug === slug) ?? null;
}

export function presetPath(
  kind: PresetKind,
  name?: string,
): `/settings/presets/${string}` {
  return name == null
    ? `/settings/presets/${kind.slug}`
    : `/settings/presets/${kind.slug}/${encodeURIComponent(name)}`;
}
