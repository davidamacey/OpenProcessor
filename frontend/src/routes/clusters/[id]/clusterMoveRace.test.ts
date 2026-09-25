/**
 * Regression test for the one-class-to-another drag-drop "flicker back and stay"
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
import { buildGroups, createGridGroups } from '$lib/gridGroups.svelte';
import { extractBalanced, extractFunction, normalize } from '$lib/testing/sourceScan';

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

/**
 * 2026-09-12 follow-up. The exclusion set above only guards the *pager*.
 * The grid the operator actually looks at is `gridGroups`, and that was a
 * separate `$state` snapshot which svelte-dnd-action's consider/finalize
 * handlers wrote their own (pre-drop) copy of the zone list into. So a drop
 * could leave the pager perfectly correct and the grid still painting the
 * moved crops in their old slots, forever — the exclusion set never sees a
 * dnd event, and the rebuild `$effect` only re-runs when the pager list
 * changes, which it doesn't after the drop handler has already run.
 *
 * Live evidence this actually happened: crops 53dd6852… and 72938398… were
 * batch-labelled cluster 15 -> 72 together at 04:57:22, and then labelled
 * 72 -> 72 *again*, individually, at 04:58:26 and 04:58:32
 * (`class_id_history` in the items index). The second write is only
 * explicable as the operator re-dragging crops the grid was still showing.
 * Instrumented drops confirm the shape: `finalize` on the origin zone
 * arrives carrying N+1 items while `cropPager.items` already holds N.
 */
interface Crop {
  id: string;
  cluster_subid: string | null;
}
const crops = (...specs: string[]): Crop[] =>
  specs.map((s) => {
    const [id, sub] = s.split(':');
    return { id: id!, cluster_subid: sub ?? null };
  });

describe('mechanism: the grid can never render a crop the pager no longer holds', () => {
  // Wired the same way the page is: the real `createPager` owns the crop
  // buffer (its `items` is the reactive source of truth) and the real
  // `createGridGroups` derives the rendered grid off it.
  async function harness(initial: Crop[], grouped = false) {
    const pager = createPager<Crop>({
      fetchPage: async () => ({ items: initial, total: initial.length }),
      keyOf: (c) => c.id,
    });
    await pager.loadFirst();
    const grid = createGridGroups<Crop>({
      source: () => pager.items,
      grouped: () => grouped,
      keyOf: (c) => c.id,
      subidOf: (c) => c.cluster_subid,
      liveIds: () => new Set(pager.items.map((c) => c.id)),
    });
    return {
      grid,
      get rendered() {
        return grid.groups.flatMap((g) => g.items.map((c) => c.id));
      },
      remove(...ids: string[]) {
        const gone = new Set(ids);
        pager.items = pager.items.filter((c) => !gone.has(c.id));
      },
    };
  }

  it('a finalize carrying the dnd library’s pre-drop list cannot resurrect the moved crops', async () => {
    const h = await harness(crops('a', 'b', 'c', 'd'));
    const preDrop = [...h.grid.groups[0]!.items];
    expect(preDrop.map((c) => c.id)).toEqual(['a', 'b', 'c', 'd']);

    // Drag 'b' + 'c' onto a class row. The ClassSidebar finalize fires first
    // and the drop handler removes BOTH optimistically...
    h.remove('b', 'c');
    expect(h.rendered).toEqual(['a', 'd']);

    // ...then the origin zone's finalize arrives with the library's stale
    // list (everything except the single dragged shadow item 'b').
    h.grid.setZoneItems(
      '__all__',
      preDrop.filter((c) => c.id !== 'b'),
    );

    // 'c' must NOT come back. Pre-fix this wrote straight into the snapshot.
    expect(h.rendered).toEqual(['a', 'd']);
  });

  it('a stale consider after the drop cannot resurrect either, and reset() restores derived truth', async () => {
    const h = await harness(crops('a', 'b', 'c'));
    const preDrop = [...h.grid.groups[0]!.items];
    h.remove('b');
    // consider has no "rebuild from the source list" line at all pre-fix --
    // whatever it wrote was the grid's last word until a hard reload.
    h.grid.setZoneItems('__all__', preDrop);
    expect(h.rendered).toEqual(['a', 'c']);
    h.grid.reset();
    expect(h.rendered).toEqual(['a', 'c']);
    expect(h.grid.overridden).toBe(false);
  });

  it('the drag-local override is transient: dropping it re-derives from the pager', async () => {
    const h = await harness(crops('a', 'b', 'c'));
    // Mid-drag reorder is honoured while the drag is live...
    h.grid.setZoneItems('__all__', crops('c', 'a', 'b'));
    expect(h.rendered).toEqual(['c', 'a', 'b']);
    expect(h.grid.overridden).toBe(true);
    // ...and discarded at drag end, so no snapshot can outlive the gesture.
    h.grid.reset();
    expect(h.rendered).toEqual(['a', 'b', 'c']);
  });

  it('a later pager change still repaints even if a drag left an override behind', async () => {
    const h = await harness(crops('a', 'b', 'c'));
    h.grid.setZoneItems('__all__', crops('a', 'b', 'c'));
    h.remove('b');
    h.grid.reset();
    expect(h.rendered).toEqual(['a', 'c']);
  });

  it('reconciliation also drops duplicate ids (a repeated key crashes the keyed each block)', async () => {
    const h = await harness(crops('a', 'b'));
    h.grid.setZoneItems('__all__', crops('a', 'b', 'b'));
    expect(h.rendered).toEqual(['a', 'b']);
  });

  it('sub-cluster grouping keeps contiguous runs and unique keys', () => {
    const groups = buildGroups(
      crops('a:15b', 'b:15a', 'c:15b', 'd'),
      true,
      (c) => c.cluster_subid,
    );
    expect(groups.map((g) => g.key)).toEqual(['15a#0', '15b#1', '__none__#2']);
    expect(new Set(groups.map((g) => g.key)).size).toBe(groups.length);
    expect(groups.at(-1)!.label).toBe('unrefined');
  });

  it('the OLD snapshot mechanism does resurrect (proves this is a real bug, not a tautology)', () => {
    // Faithful model of the pre-fix page: gridGroups is a plain snapshot, the
    // rebuild effect only runs when the source list changes, and the dnd
    // handlers assign their own payload into it.
    const source = crops('a', 'b', 'c', 'd');
    let snapshot = buildGroups(source, false, (c) => c.cluster_subid);
    const preDrop = [...snapshot[0]!.items];

    // Drop handler removes b + c from the pager; the effect reruns.
    const afterDrop = source.filter((c) => c.id !== 'b' && c.id !== 'c');
    snapshot = buildGroups(afterDrop, false, (c) => c.cluster_subid);
    expect(snapshot.flatMap((g) => g.items.map((c) => c.id))).toEqual(['a', 'd']);

    // Then a dnd event lands with the library's stale list and wins, with no
    // source change left to re-trigger the rebuild effect.
    snapshot[0]!.items = preDrop.filter((c) => c.id !== 'b');
    expect(snapshot.flatMap((g) => g.items.map((c) => c.id))).toEqual(['a', 'c', 'd']);
    //                                                                       ^ 'c' is back
  });
});

// P2-2 (docs/design/test-audit-2026-09-24.md T1): every extraction below
// went through `$lib/testing/sourceScan.ts`'s `extractFunction`/
// `extractBalanced` (brace-balanced) or `normalize` (comment-stripped,
// whitespace-collapsed) instead of a `\n {2}\}`/`\n {4}\}?\);`-anchored
// regex, which broke on a harmless prettier re-wrap or indentation-width
// change — verified against a mutated /tmp copy that reformatted
// `onGroupFinalize` and inserted a double space in the
// `excludedCropIds` declaration; both still pass here, and dropping the
// `excludedCropIds.add` call still fails.
const normalizedSrc = normalize(src);

describe('wiring: /clusters/[id] +page.svelte derives the grid instead of snapshotting it', () => {
  it('gridGroups is a $derived off createGridGroups, not a writable $state snapshot', () => {
    expect(normalizedSrc).toMatch(/const gridGroups = \$derived\(grid\.groups\)/);
    expect(normalizedSrc).not.toMatch(/let gridGroups = \$state/);
    // The old rebuild effect and its hand-rolled snapshot writer are gone.
    expect(normalizedSrc).not.toMatch(/gridGroups = buildGroups\(/);
    expect(normalizedSrc).not.toMatch(/function _setGroupItems/);
  });

  it('the grid state is told what is still live so a dnd event cannot resurrect a moved crop', () => {
    const wiring = extractBalanced(src, /const grid = createGridGroups<Crop>\(\{/);
    expect(wiring).not.toBeNull();
    expect(normalize(wiring!)).toMatch(
      /liveIds: \(\) => new Set\(cropPager\.items\.map\(\(c\) => c\.id\)\)/,
    );
  });

  it('onGroupFinalize discards the drag override rather than adopting the library’s stale list', () => {
    const fn = extractFunction(src, 'onGroupFinalize');
    expect(fn).not.toBeNull();
    expect(normalize(fn!)).toMatch(/grid\.reset\(\)/);
    // Strip comments before asserting the payload is never adopted — the
    // comment above the reset() names e.detail.items on purpose.
    // normalize() already strips comments; check against it directly.
    expect(normalize(fn!)).not.toMatch(/setZoneItems/);
    expect(normalize(fn!)).not.toMatch(/e\.detail\.items/);
  });
});

// The `describe('wiring: ... actually uses the exclusion set', ...)` block
// that used to live here (source-scanning +page.svelte for
// `excludedCropIds.add`/`.delete` call sites across the sidebar-drop
// handler, moveCropIds, ignoreSelected/undoIgnore, the D hotkey handler,
// and undoLast) is deleted per docs/design/test-audit-2026-09-24.md P1-4
// (second pass): that logic — and the exclusion set itself, now
// `ExclusionGuard` — moved into `$lib/clusters/clusterController.svelte.ts`
// (commit history: "refactor(clusters): extract action logic into
// clusterController"). `clusterController.test.ts` exercises the same
// guarantees behaviorally instead of by regex:
//   - 'claims dropped ids in the exclusion guard before the request
//     settles, and resets the grid' / 'releases conflicted ids ...' /
//     'reverts the optimistic removal and releases claimed ids ...'
//     (describe('handleClassDrop')) replace the old sidebar-drop-handler
//     scan (`the sidebar-drop (dropOnClassStore) handler claims dragged
//     ids ...` and the "resets the grid override" test that used to live
//     in the describe block above, both here originally);
//   - the three equivalently-named tests in describe('moveCropIds')
//     replace 'moveCropIds (M hotkey / move picker) claims ids ...';
//   - describe('ignoreSelected / undoIgnore') replaces 'ignoreSelected
//     claims ids and undoIgnore releases them';
//   - describe('discardSelected then undoLast')'s first test (checks
//     `exclusionGuard.accept(a)` is false after a claim) replaces 'the
//     discard (D) hotkey handler claims successfully-discarded ids';
//   - describe('undoLast branch coverage') replaces 'undoLast releases
//     the restored crop id back so it can reappear' — and additionally
//     exercises both branches of the `if (cropPager.items.some(...))`
//     check that survived as an `if (false)` mutant under the old scan
//     (docs/design/test-audit-2026-09-24.md §2.2).
// The remaining `excludedCropIds`-named text above (a page-local
// `const excludedCropIds = new Set<string>()` and its `accept:` wiring)
// no longer exists in +page.svelte at all — the page now owns
// `exclusionGuard` (from `createExclusionGuard()`) and passes
// `exclusionGuard.accept` straight into `cropPager`'s `accept` option, so
// there was nothing left in the page source for a scan to check.
