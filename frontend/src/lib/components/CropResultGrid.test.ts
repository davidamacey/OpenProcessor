/**
 * `<CropResultGrid>` extracts the flat dndzone + selection + CropCard/
 * ScoreChip overlay pattern that previously lived inline in
 * `/clusters/[id]` (see that route's `onGroupConsider`/`onGroupFinalize`
 * for the pattern this mirrors: capture the multi-drag set on
 * consider, reconcile/drop the drag-local override on finalize so a
 * post-drop dnd snapshot can never repaint an item the caller already
 * removed). No `@testing-library/svelte` here (see StrategyBar.test.ts),
 * so this is a static source scan of the same load-bearing behaviors.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));

function read(rel: string): string {
  return readFileSync(path.resolve(here, rel), 'utf-8');
}

describe('CropResultGrid.svelte', () => {
  const src = read('./CropResultGrid.svelte');

  it('is a single flat dndzone — no per-sub-cluster grouping loop', () => {
    const matches = src.match(/use:dndzone/g) ?? [];
    expect(matches.length).toBe(1);
  });

  it('rejects drops from other zones (dropFromOthersDisabled) same as /clusters/[id]', () => {
    expect(src).toMatch(/dropFromOthersDisabled:\s*true/);
  });

  it('captures the Finder-pattern multi-drag set on consider (drag a selected card drags the whole selection)', () => {
    expect(src).toMatch(/sel\.has\(draggedId\)\s*&&\s*sel\.size > 1/);
  });

  it('drops the drag-local override on finalize rather than adopting the dnd library snapshot', () => {
    expect(src).toMatch(/dragItems = null/);
  });

  it('renders the match-score chip only when scoreOf returns non-null', () => {
    expect(src).toMatch(/\{#if scoreOf\}/);
    expect(src).toMatch(/score != null/);
  });

  it('exposes a cornerBadge snippet slot for the caller-supplied overlay (e.g. cluster-origin badge)', () => {
    expect(src).toMatch(/cornerBadge/);
    expect(src).toMatch(/@render cornerBadge/);
  });

  it('p7 (2026-09-24 interactive pass): renders a top scrim behind the chip row whenever either chip can appear', () => {
    // Must be gated on the SAME condition the two chips use
    // (scoreOf?.(crop) != null || cornerBadge), not a narrower one that
    // would leave a chip unscrimmed.
    expect(src).toMatch(/\{#if scoreOf\?\.\(crop\) != null \|\| cornerBadge\}/);
    // C6 (visual audit 2026-09-24): a flat scrim, never a gradient.
    expect(src).toMatch(/top-0 h-7 rounded-t-md bg-black\/55/);
    expect(src).not.toMatch(/bg-gradient/);
  });
});
