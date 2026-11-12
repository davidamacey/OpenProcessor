/**
 * Role-driven detector chip color resolution.
 *
 * Detector/verifier/segmenter LABELS now come from the backend's served
 * vocabulary (`GET {API_PREFIX}/regions/vocabulary` —
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
import type { RegionVocabularyRole } from '$lib/api_regions';

export const PALETTES = {
  blue: { border: 'border-accent/50', bg: 'bg-accent/10', text: 'text-accent-text' },
  purple: {
    border: 'border-holdout/50',
    bg: 'bg-holdout/10',
    text: 'text-holdout-text',
  },
  amber: { border: 'border-warn/50', bg: 'bg-warn/10', text: 'text-warn-text' },
  emerald: {
    border: 'border-success/50',
    bg: 'bg-success/10',
    text: 'text-success-text',
  },
  teal: { border: 'border-info/50', bg: 'bg-info/10', text: 'text-info-text' },
  rose: { border: 'border-danger/50', bg: 'bg-danger/10', text: 'text-danger-text' },
  sky: { border: 'border-info/50', bg: 'bg-info/10', text: 'text-info-text' },
  indigo: {
    border: 'border-accent/50',
    bg: 'bg-accent/10',
    text: 'text-accent-text',
  },
  orange: {
    border: 'border-warn/50',
    bg: 'bg-warn/10',
    text: 'text-warn-text',
  },
  zinc: { border: 'border-line-strong', bg: 'bg-surface-2/60', text: 'text-fg-2' },
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
