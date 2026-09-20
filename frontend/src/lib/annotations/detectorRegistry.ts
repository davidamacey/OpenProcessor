/**
 * Config-driven detector label/palette resolution.
 *
 * `DetectorChip.svelte`'s `labelFor()`/`paletteFor()` are a hand-written
 * switch/if-chain over legacy's specific detector ids. This module is
 * the generalized, data-driven replacement: `legacyDetectors.ts` (the
 * profile) reproduces those two functions' exact resolution — including
 * palette-lookup order, since it's semantically load-bearing (an exact
 * match must be checked before prefix rules) — as data, and a future
 * deployment can swap in its own registry with no component change.
 *
 * Not yet wired into `DetectorChip.svelte` — see
 * docs/genericization-plan-2026-09-13.md §3.2 / P2.2 for that migration
 * (ships with a 21-id-plus-unknown-plus-null equivalence snapshot test
 * proving the swap is pixel-identical). This module and its equivalence
 * test (`legacyDetectors.test.ts`) are that migration's prerequisite.
 */

import type { Palette } from './types';

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

export interface DetectorRegistry {
  /** Exact detector id → short label. Falls through to the raw id when
   *  absent, so a newly-deployed detector is never silently swallowed. */
  labels: Record<string, string>;
  /** Exact detector id → palette. Evaluated BEFORE `prefixes`. */
  palettes: Record<string, PaletteName>;
  /** Ordered prefix rules. FIRST MATCH WINS — order is behavior. */
  prefixes: Array<{ startsWith: string; palette: PaletteName }>;
  fallback: PaletteName;
  /** Tags rendered at reduced opacity. */
  mutedTagPattern: RegExp;
}

export function labelForDetector(registry: DetectorRegistry, d: string | null): string {
  if (!d) return '—';
  return registry.labels[d] ?? d;
}

export function paletteForDetector(
  registry: DetectorRegistry,
  d: string | null,
): Palette {
  if (!d) return PALETTES[registry.fallback];
  const exact = registry.palettes[d];
  if (exact) return PALETTES[exact];
  const prefixMatch = registry.prefixes.find((p) => d.startsWith(p.startsWith));
  if (prefixMatch) return PALETTES[prefixMatch.palette];
  return PALETTES[registry.fallback];
}

export function isMutedTag(registry: DetectorRegistry, tag: string | null): boolean {
  if (!tag) return false;
  return registry.mutedTagPattern.test(tag);
}
