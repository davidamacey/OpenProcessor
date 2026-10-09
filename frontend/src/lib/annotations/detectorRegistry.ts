/**
 * Role-driven detector chip color resolution.
 *
 * Detector/verifier/segmenter LABELS now come from the backend's served
 * vocabulary (`GET {API_PREFIX}/regions/vocabulary`, W0 naming-sweep finding m9 —
 * see `$stores/regionVocabulary.svelte`), not from a hand-copied
 * name→label table: those model ids are deployment config, and a
 * hardcoded map drifted from them the moment a deployment swapped
 * detectors. Chip COLOR is still a purely-display concern this module
 * owns — it's keyed off the vocabulary entry's `role`
 * (`detector | segmenter | ocr | verifier | human | classifier |
 * proposal`), not the id, so a new deployment's detector automatically
 * gets a sensible color the moment the backend reports its role.
 *
 * `mutedTagPattern` (outcome/tag business logic, not a naming table)
 * still lives per-deployment in `builtinDetectors.ts`.
 */

import type { Palette } from './types';
import type { RegionVocabularyRole } from '$lib/api';

export const PALETTES = {
  blue: { border: 'border-blue-500/50', bg: 'bg-blue-500/10', text: 'text-blue-200' },
  purple: {
    border: 'border-purple-500/50',
    bg: 'bg-purple-500/10',
    text: 'text-purple-200',
  },
  amber: { border: 'border-amber-500/50', bg: 'bg-amber-500/10', text: 'text-amber-200' },
  emerald: {
    border: 'border-emerald-500/50',
    bg: 'bg-emerald-500/10',
    text: 'text-emerald-200',
  },
  teal: { border: 'border-teal-500/50', bg: 'bg-teal-500/10', text: 'text-teal-200' },
  rose: { border: 'border-rose-500/50', bg: 'bg-rose-500/10', text: 'text-rose-200' },
  sky: { border: 'border-sky-500/50', bg: 'bg-sky-500/10', text: 'text-sky-200' },
  indigo: {
    border: 'border-indigo-500/50',
    bg: 'bg-indigo-500/10',
    text: 'text-indigo-200',
  },
  orange: {
    border: 'border-orange-500/50',
    bg: 'bg-orange-500/10',
    text: 'text-orange-200',
  },
  zinc: { border: 'border-zinc-700', bg: 'bg-zinc-800/60', text: 'text-zinc-300' },
} as const satisfies Record<string, Palette>;

export type PaletteName = keyof typeof PALETTES;

/** Role → palette. Every role the vocabulary contract documents gets an
 *  entry; an unrecognized role (or no role at all — an id the vocabulary
 *  doesn't know about) falls back to the neutral `zinc` chip. */
const ROLE_PALETTES: Record<string, PaletteName> = {
  detector: 'blue',
  segmenter: 'purple',
  ocr: 'amber',
  verifier: 'teal',
  human: 'emerald',
  classifier: 'indigo',
  proposal: 'sky',
};

export function paletteForRole(role: RegionVocabularyRole | null | undefined): Palette {
  if (!role) return PALETTES.zinc;
  const name = ROLE_PALETTES[role];
  return name ? PALETTES[name] : PALETTES.zinc;
}

/** Tags rendered at reduced opacity — outcome/business logic, not a
 *  naming table, so it stays a small per-deployment config
 *  (`builtinDetectors.ts`) rather than served vocabulary. */
export interface MutedTagConfig {
  mutedTagPattern: RegExp;
}

export function isMutedTag(config: MutedTagConfig, tag: string | null): boolean {
  if (!tag) return false;
  return config.mutedTagPattern.test(tag);
}
