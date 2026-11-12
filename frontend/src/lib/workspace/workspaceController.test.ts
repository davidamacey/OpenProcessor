import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';

const nav = vi.hoisted(() => ({ goto: vi.fn(), getCrop: vi.fn() }));
vi.mock('$app/state', async () => await import('$lib/test/pageState.svelte'));
vi.mock('$app/navigation', () => ({ goto: nav.goto }));
vi.mock('$lib/api_cropRead', () => ({ getCrop: nav.getCrop }));
vi.mock('$app/paths', () => ({ resolve: (p: string) => p }));
vi.mock('$lib/projectPaths', () => ({ projectHref: (p: string) => `/p/default${p}` }));

import { visit as visitUrl } from '$lib/test/pageState.svelte';
import { nextStepsStore } from '$stores/nextSteps.svelte';
import {
  createWorkspaceController,
  type WorkspaceController,
} from './workspaceController.svelte';

let ws: WorkspaceController | null = null;

function visit(search: string): void {
  visitUrl(`/p/default/workspace${search}`);
}

function make(): WorkspaceController {
  ws = createWorkspaceController();
  flushSync();
  return ws;
}

function lastGoto(): { href: string; opts: Record<string, unknown> } {
  const [href, opts] = nav.goto.mock.calls.at(-1)!;
  return { href, opts };
}

beforeEach(() => {
  nav.goto.mockReset();
  visit('');
  nextStepsStore.reset();
});

afterEach(() => {
  ws?.dispose();
  ws = null;
});

describe('WorkspaceController', () => {
  it('switching views keeps filter, queue, scope and selection', () => {
    visit('?view=detections&class_name=widget&queue=to_label&scope=cluster%3Au1');
    const c = make();
    c.select({ kind: 'items', cropIds: ['c1'] });
    const selection = c.selection;
    const before = JSON.stringify(c.itemQuery());

    c.setView('table');
    const { href, opts } = lastGoto();
    visit(href.slice(href.indexOf('?')));
    flushSync();

    expect(c.view).toBe('table');
    expect(JSON.stringify(c.itemQuery())).toBe(before);
    expect(c.url.scope).toEqual({ kind: 'cluster', id: 'u1' });
    expect(c.selection).toBe(selection);
    expect(opts).toMatchObject({ replace: false, reset: false });
  });

  it('switching to a view with another selection kind clears the selection', () => {
    visit('?view=clusters');
    const c = make();
    c.select({ kind: 'clusters', clusterUids: ['u1'] });
    c.setView('detections');
    const { href } = lastGoto();
    visit(href.slice(href.indexOf('?')));
    flushSync();
    expect(c.selection.kind).toBe('none');
  });

  it('keeps an all-matching selection across views', () => {
    visit('?view=clusters');
    const c = make();
    c.select({ kind: 'matching' });
    visit('?view=detections');
    flushSync();
    expect(c.selection.kind).toBe('matching');
  });

  it('parks a focus the next view cannot show, and restores it where it fits', () => {
    visit('?view=detections&focus=item%3Acrop-1');
    const c = make();
    c.setView('map');
    const first = new URLSearchParams(lastGoto().href.split('?')[1]);
    expect(first.get('view')).toBe('map');
    expect(first.has('focus')).toBe(false);
    visit(`?${first.toString()}`);
    flushSync();
    c.setView('table');
    const second = new URLSearchParams(lastGoto().href.split('?')[1]);
    expect(second.get('focus')).toBe('item:crop-1');
  });

  it('keeps a focus the next view can show', () => {
    visit('?view=detections&focus=item%3Acrop-1');
    const c = make();
    c.setView('table');
    expect(new URLSearchParams(lastGoto().href.split('?')[1]).get('focus')).toBe(
      'item:crop-1',
    );
  });

  it('closeReview restores a focus parked when Review was left', () => {
    visit('?view=review&focus=item%3Acrop-1');
    const c = make();
    c.setView('gallery');
    const params = new URLSearchParams(lastGoto().href.split('?')[1]);
    expect(params.has('focus')).toBe(false);
    visit(`?${params.toString()}`);
    flushSync();
    c.closeReview();
    expect(new URLSearchParams(lastGoto().href.split('?')[1]).get('focus')).toBe(
      'item:crop-1',
    );
  });

  it('back restores the previous view and filter', () => {
    visit('?view=clusters&class_name=widget');
    const c = make();
    expect(c.view).toBe('clusters');
    expect(c.itemQuery().class_name).toEqual(['widget']);

    visit('?view=detections&class_name=gadget');
    flushSync();
    expect(c.view).toBe('detections');
    expect(c.itemQuery().class_name).toEqual(['gadget']);
    expect(c.filter.classNames).toEqual(['gadget']);

    visit('?view=clusters&class_name=widget');
    flushSync();
    expect(c.view).toBe('clusters');
    expect(c.itemQuery().class_name).toEqual(['widget']);
    expect(c.filter.classNames).toEqual(['widget']);
  });

  it('uses the served default view when the URL has none', () => {
    nextStepsStore.defaultView = 'clusters';
    const c = make();
    expect(c.view).toBe('clusters');
    expect(nav.goto).toHaveBeenCalledTimes(1);
    const { href, opts } = lastGoto();
    expect(href).toBe('/p/default/workspace?view=clusters');
    expect(opts.replace).toBe(true);
  });

  it('falls back to the gallery before the default is served, without a redirect', () => {
    const c = make();
    expect(c.view).toBe('gallery');
    expect(nav.goto).not.toHaveBeenCalled();
  });

  it('keeps an explicit view over the served default', () => {
    nextStepsStore.defaultView = 'clusters';
    visit('?view=gallery');
    const c = make();
    expect(c.view).toBe('gallery');
    expect(nav.goto).not.toHaveBeenCalled();
  });

  it('the queue reaches every view as review_tab', () => {
    const c = make();
    c.setQueue('to_label');
    const { href } = lastGoto();
    visit(href.slice(href.indexOf('?')));
    flushSync();
    expect(c.itemQuery().review_tab).toBe('to_label');
    expect(c.itemBody().review_tab).toBe('to_label');
  });

  it('another queue starts in its own order: the Review sort is dropped, a Detections sort stays', () => {
    visit(
      '?view=review&queue=to_label&sort_review=atypicality&sort_detections=class_name%3Aasc',
    );
    const c = make();
    c.setQueue('check_machine');
    const params = new URLSearchParams(lastGoto().href.split('?')[1]);
    expect(params.get('queue')).toBe('check_machine');
    expect(params.has('sort_review')).toBe(false);
    expect(params.get('sort_detections')).toBe('class_name:asc');
  });

  it('two URL writes in one tick both land (the second starts from the first)', () => {
    visit('?view=review&queue=to_label');
    const c = make();
    c.patchUrl((p) => p.set('q', 'red widget'));
    c.setReviewFocus('crop-7');
    const params = new URLSearchParams(lastGoto().href.split('?')[1]);
    expect(params.get('q')).toBe('red widget');
    expect(params.get('focus')).toBe('item:crop-7');
  });

  it('cluster scope adds the cluster to the filter', () => {
    const id = '0c6f5a0e-1111-4222-8333-444455556666';
    const c = make();
    c.setScope({ kind: 'cluster', id });
    const { href } = lastGoto();
    visit(href.slice(href.indexOf('?')));
    flushSync();
    expect(c.itemBody().cluster_uids).toEqual([id]);
    expect(c.itemQuery().cluster_uid).toEqual([id]);
  });

  it('names a near-duplicate scope for the bulk bar', () => {
    const c = make();
    expect(c.inNearDupGroup).toBe(false);
    c.setScope({ kind: 'near_dup', id: 'g1' });
    const { href } = lastGoto();
    visit(href.slice(href.indexOf('?')));
    flushSync();
    expect(c.inNearDupGroup).toBe(true);
  });

  it('an unapplied draft does not change the query', () => {
    visit('?class_name=widget');
    const c = make();
    c.filter.classNames = ['gadget'];
    expect(c.itemQuery().class_name).toEqual(['widget']);
    c.applyFilter();
    expect(lastGoto().href).toContain('class_name=gadget');
  });

  it('removing a chip writes the URL without that value', () => {
    visit('?class_name=widget&class_name=gadget&view=gallery');
    const c = make();
    c.removeChip('class_name', 'widget');
    const { href } = lastGoto();
    expect(href).toContain('class_name=gadget');
    expect(href).not.toContain('class_name=widget');
    expect(href).toContain('view=gallery');
  });

  it('builds the selection body from the current selection kind', () => {
    visit('?view=clusters&class_name=widget');
    const c = make();
    c.select({ kind: 'clusters', clusterUids: ['u1'] });
    expect(c.itemSelection()).toEqual({
      filter: { class_names: ['widget'], cluster_uids: ['u1'] },
    });
  });

  it('select-all goes through the registered page; a second press on a full page asks for all matching', () => {
    visit('?view=detections');
    const c = make();
    const applied: string[][] = [];
    c.registerPage({
      kind: 'items',
      ids: () => ['a', 'b'],
      total: () => 50,
      apply: (ids) => applied.push(ids),
    });
    expect(c.pressSelectAll()).toBe(true);
    expect(applied).toEqual([['a', 'b']]);
    expect(c.selection).toEqual({ kind: 'items', cropIds: ['a', 'b'] });
    expect(c.pendingSelectAll).toBeNull();
    expect(c.pressSelectAll()).toBe(true);
    expect(c.pendingSelectAll).toEqual({ total: 50 });
    expect(applied).toHaveLength(1);
    c.confirmSelectAllMatching();
    expect(c.selection).toEqual({ kind: 'matching' });
    expect(c.pendingSelectAll).toBeNull();
  });

  it('select-all uses the kind the page registers, not the view', () => {
    visit('?view=clusters&scope=cluster%3Au1');
    const c = make();
    c.registerPage({ kind: 'items', ids: () => ['a'], total: () => 1, apply: () => {} });
    c.pressSelectAll();
    expect(c.selection).toEqual({ kind: 'items', cropIds: ['a'] });
  });

  it('select-all is not handled while no page is registered', () => {
    const c = make();
    expect(c.pressSelectAll()).toBe(false);
    const off = c.registerPage({
      kind: 'items',
      ids: () => [],
      total: () => 0,
      apply: () => {},
    });
    off();
    expect(c.pressSelectAll()).toBe(false);
  });

  it('a view can drop filter keys it does not apply', () => {
    visit('?class_name=widget&max_rank=2');
    const c = make();
    expect(c.itemQuery().max_rank).toBe(2);
    expect(c.itemQuery((p) => p !== 'max_rank').max_rank).toBeUndefined();
  });

  it('patchUrl rewrites the keys a view owns and keeps the rest', () => {
    visit('?view=clusters&class_name=widget&sort=size_desc');
    const c = make();
    c.patchUrl((p) => p.set('sort', 'updated_desc'));
    const { href, opts } = lastGoto();
    expect(href).toContain('sort=updated_desc');
    expect(href).toContain('view=clusters');
    expect(href).toContain('class_name=widget');
    expect(opts.replace).toBe(true);
  });

  it('matching mode is on for mode=matching or an open-vocabulary filter', () => {
    visit('?view=clusters&mode=matching');
    expect(make().matchingMode).toBe(true);
    ws?.dispose();
    visit('?view=clusters&open_vocab_set=widgets');
    expect(make().matchingMode).toBe(true);
    ws?.dispose();
    visit('?view=clusters');
    expect(make().matchingMode).toBe(false);
  });

  it('leaving matching mode drops the mode and the open-vocabulary pair', () => {
    visit('?view=clusters&mode=matching&open_vocab_set=widgets&source_prompt=w');
    const c = make();
    c.setMatchingMode(false);
    const { href } = lastGoto();
    expect(href).not.toContain('mode=');
    expect(href).not.toContain('open_vocab_set');
    expect(href).not.toContain('source_prompt');
  });

  it('open review keeps the focus and returns to it', () => {
    visit('?view=detections&class_name=widget&focus=item%3Aa3');
    const c = make();

    c.openReview('a3');
    const opened = lastGoto();
    expect(opened.href).toContain('view=review');
    expect(opened.href).toContain('focus=item%3Aa3');
    expect(opened.href).toContain('class_name=widget');
    expect(opened.opts).toMatchObject({ replace: false, reset: false });

    // Review walks on to the next unhandled item and records it as the focus.
    visit(opened.href.slice(opened.href.indexOf('?')));
    flushSync();
    c.setReviewFocus('a4');
    expect(lastGoto().opts).toMatchObject({ replace: true });
    expect(c.view).toBe('review');

    // `Esc` before that URL write has landed still returns to the card Review stopped on.
    c.closeReview();
    const back = lastGoto();
    expect(back.href).toContain('view=detections');
    expect(back.href).toContain('focus=item%3Aa4');
    expect(back.href).toContain('class_name=widget');
  });

  it('closing a Review that was deep-linked returns to the Detections grid', () => {
    visit('?view=review&queue=to_label&focus=item%3Ab1');
    const c = make();
    c.closeReview();
    expect(lastGoto().href).toContain('view=detections');
    expect(lastGoto().href).toContain('queue=to_label');
    expect(lastGoto().href).toContain('focus=item%3Ab1');
  });

  it('opens the focused image in the editor, keeping the Workspace query', async () => {
    visit('?view=gallery&class_name=widget&focus=image%3Aimg-7');
    const c = make();
    await c.openEditor();
    expect(lastGoto().href).toBe(
      '/p/default/images/img-7?view=gallery&class_name=widget&focus=image%3Aimg-7',
    );
    expect(nav.getCrop).not.toHaveBeenCalled();
  });

  it("opens the focused item's image in the editor", async () => {
    nav.getCrop.mockResolvedValue({ id: 'c9', image_id: 'img-9' });
    visit('?focus=item%3Ac9');
    const c = make();
    await c.openEditor();
    expect(nav.getCrop).toHaveBeenCalledWith('c9');
    expect(lastGoto().href).toBe('/p/default/images/img-9?focus=item%3Ac9');
  });

  it('opens nothing when nothing with an image has focus', async () => {
    visit('?focus=cluster%3Au1');
    const c = make();
    await c.openEditor();
    expect(nav.goto).not.toHaveBeenCalled();
  });

  describe('lightbox', () => {
    it('opens on the focused image and closes', async () => {
      visit('?view=gallery&focus=image%3Aimg-2');
      const c = make();
      await c.openLightbox();
      expect(c.lightbox?.imageId).toBe('img-2');
      c.closeLightbox();
      expect(c.lightbox).toBeNull();
    });

    it("opens on a focused item's image", async () => {
      nav.getCrop.mockResolvedValue({ image_id: 'img-9' });
      visit('?view=detections&focus=item%3Acrop-1');
      const c = make();
      await c.openLightbox();
      expect(c.lightbox?.imageId).toBe('img-9');
    });

    it('steps through the images the gallery has loaded and stops at the ends', async () => {
      visit('?view=gallery');
      const c = make();
      c.registerPage({
        kind: 'images',
        ids: () => ['a', 'b', 'c'],
        total: () => 3,
        apply: () => {},
      });
      await c.openLightbox('b');
      expect(c.lightboxNeighbour(-1)).toBe('a');
      c.stepLightbox(1);
      expect(c.lightbox?.imageId).toBe('c');
      expect(c.lightboxNeighbour(1)).toBeNull();
      c.stepLightbox(1);
      expect(c.lightbox?.imageId).toBe('c');
    });

    it('has no neighbours where the view lists no images', async () => {
      const c = make();
      await c.openLightbox('x');
      expect(c.lightboxNeighbour(1)).toBeNull();
    });
  });
});
