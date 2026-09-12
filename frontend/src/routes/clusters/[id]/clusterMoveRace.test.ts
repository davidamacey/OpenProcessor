/**
 * Regression test for the cobalt->subaru drag-drop "flicker back and stay"
 * bug (2026-09-11 live report): dragging crops out of a cluster onto
 * another class in the sidebar (or via the M move-picker / discard / ignore
 * paths) removed them from `cropPager.items` optimistically, but nothing
 * stopped a same-cluster GET already in flight — or freshly triggered by
 * the SSE live-refresh `$effect` a few lines below — from resolving with a
 * pre-move snapshot and overwriting that optimistic removal outright. The
 * crop would flicker away, then reappear and *stay*, because nothing ever
 * re-asserted the move; only a hard reload (which finally reads
 * post-write server state) made it disappear again. The move itself was
 * never wrong server-side (`batch_label`/`move_crops` both write with
 * `refresh=True`) — this was purely a frontend stale-response race.
 *
 * The fix adds a page-local `excludedCropIds` set: every optimistic
 * removal claims its ids in the set *before* the awaited request settles,
 * and `cropPager`'s `accept` filter (re-evaluated against the set's
 * *current* contents at fetch-resolution time, not fetch-start time) is
 * wired to strip anything claimed — so a stale response can never
 * resurrect a crop the operator already moved away. ids are released
 * back (`excludedCropIds.delete`) on revert/undo so a crop that never
 * actually left is never permanently hidden.
 *
 * This repo has no `@testing-library/svelte` (see StrategyBar.test.ts /
 * TrainForm.test.ts for the established precedent), so half of this is a
 * static source scan of `+page.svelte` confirming every mutation site
 * actually claims/releases ids — that part would have failed against the
 * pre-fix source (grep for `excludedCropIds` return nothing there). The
 * other half exercises the real, unmocked `createPager` to prove the
 * exclusion-set + `accept` mechanism itself actually closes the race,
 * not just that the words appear in the source.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { createPager } from '$lib/pager.svelte';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, './+page.svelte'), 'utf-8');

interface Row {
  id: string;
}
const rows = (...ids: string[]): Row[] => ids.map((id) => ({ id }));

describe('mechanism: exclusion set survives a stale in-flight fetch (executable, unmocked pager)', () => {
  it('a fetch already in flight when a crop is claimed must not resurrect it on resolve', async () => {
    const excluded = new Set<string>();
    let resolveFetch!: (v: { items: Row[]; total: number }) => void;
    const fetchPage = () =>
      new Promise<{ items: Row[]; total: number }>((resolve) => {
        resolveFetch = resolve;
      });
    const pager = createPager<Row>({
      fetchPage,
      keyOf: (r) => r.id,
      accept: (r) => !excluded.has(r.id),
    });

    // Analogue of the SSE live-refresh effect's loadFirst() -- issued,
    // in flight, snapshot not taken yet.
    const loadPromise = pager.loadFirst();

    // Analogue of the human's drag-drop landing *after* that GET went
    // out but *before* it comes back: optimistic removal + claim.
    pager.items = pager.items.filter((r) => r.id !== 'b');
    excluded.add('b');

    // The in-flight GET finally resolves with a pre-move snapshot that
    // still has 'b' in it.
    resolveFetch({ items: rows('a', 'b'), total: 2 });
    await loadPromise;

    expect(pager.items.map((r) => r.id)).toEqual(['a']);
  });

  it('without the exclusion claim, the same race resurrects the crop (proves this is a real race, not a tautology)', async () => {
    let resolveFetch!: (v: { items: Row[]; total: number }) => void;
    const fetchPage = () =>
      new Promise<{ items: Row[]; total: number }>((resolve) => {
        resolveFetch = resolve;
      });
    const pager = createPager<Row>({ fetchPage, keyOf: (r) => r.id }); // no `accept`

    const loadPromise = pager.loadFirst();
    pager.items = pager.items.filter((r) => r.id !== 'b'); // optimistic removal, unclaimed
    resolveFetch({ items: rows('a', 'b'), total: 2 });
    await loadPromise;

    // Bug reproduced at the primitive level: the stale response silently
    // un-does the optimistic removal.
    expect(pager.items.map((r) => r.id)).toEqual(['a', 'b']);
  });
});

describe('wiring: /clusters/[id] +page.svelte actually uses the exclusion set', () => {
  it('declares a page-local excludedCropIds set', () => {
    expect(src).toMatch(/const excludedCropIds = new Set<string>\(\)/);
  });

  it("wires cropPager's accept to the exclusion set (closes the stale-fetch race for every fetchPage call, loadFirst included)", () => {
    expect(src).toMatch(/accept:\s*\(c\)\s*=>\s*!excludedCropIds\.has\(c\.id\)/);
  });

  it('the sidebar-drop (dropOnClassStore) handler claims dragged ids before awaiting bulkLabel', () => {
    const handler = src.match(
      /dropOnClassStore\.register\(async[\s\S]*?\n {4}\}\);/,
    )?.[0];
    expect(handler).toBeDefined();
    expect(handler).toMatch(/for \(const id of ids\) excludedCropIds\.add\(id\)/);
    // Conflicted ids never actually left -- must be released before the resync.
    expect(handler).toMatch(/excludedCropIds\.delete\(c\.crop_id\)/);
    // Hard failure reverts the optimistic removal -- must release too.
    expect(handler).toMatch(/catch[\s\S]*?for \(const id of ids\) excludedCropIds\.delete\(id\)/);
  });

  it('moveCropIds (M hotkey / move picker) claims ids before awaiting moveCropsToCluster', () => {
    const fn = src.match(
      /async function moveCropIds\([\s\S]*?\n {2}\}/,
    )?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/for \(const id of ids\) excludedCropIds\.add\(id\)/);
    expect(fn).toMatch(/excludedCropIds\.delete\(c\.crop_id\)/);
    expect(fn).toMatch(/catch[\s\S]*?for \(const id of ids\) excludedCropIds\.delete\(id\)/);
  });

  it('ignoreSelected claims ids and undoIgnore releases them', () => {
    const ignoreFn = src.match(
      /async function ignoreSelected\([\s\S]*?\n {2}\}/,
    )?.[0];
    const undoIgnoreFn = src.match(
      /async function undoIgnore\([\s\S]*?\n {2}\}/,
    )?.[0];
    expect(ignoreFn).toMatch(/for \(const id of ids\) excludedCropIds\.add\(id\)/);
    expect(undoIgnoreFn).toMatch(/for \(const id of ids\) excludedCropIds\.delete\(id\)/);
  });

  it('the discard (D) hotkey handler claims successfully-discarded ids', () => {
    expect(src).toMatch(
      /cropPager\.items = cropPager\.items\.filter\(\(c\) => !succeededSet\.has\(c\.id\)\);\s*\n\s*for \(const id of succeeded\) excludedCropIds\.add\(id\)/,
    );
  });

  it('undoLast releases the restored crop id back so it can reappear', () => {
    const fn = src.match(/async function undoLast\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/excludedCropIds\.delete\(entry\.crop_id\)/);
  });
});
