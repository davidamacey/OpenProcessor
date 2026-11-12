import { describe, expect, it } from 'vitest';
import {
  clusterScopeSearch,
  parseWorkspaceUrl,
  writeWorkspaceUrl,
  WORKSPACE_URL_KEYS,
} from './workspaceUrl';

const FULL =
  'view=table&queue=to_label&scope=cluster%3Aabc&focus=item%3Acrop1' +
  '&sort_gallery=taken_desc&sort_clusters=size_desc&sort_detections=conf_asc' +
  '&sort_review=rank&sort_table=class&rows=images&columns=class%2Cconf' +
  '&thumb=l&q=red+widget&similar_to=crop9&map=12%2F51.51%2F-0.13';

const entries = (q: string | URLSearchParams): [string, string][] =>
  [...new URLSearchParams(q).entries()].sort(([a], [b]) => a.localeCompare(b));

function roundTrip(query: string): [string, string][] {
  const out = new URLSearchParams();
  writeWorkspaceUrl(out, parseWorkspaceUrl(new URLSearchParams(query)));
  return entries(out);
}

describe('workspaceUrl', () => {
  it('round-trips every parameter', () => {
    expect(roundTrip(FULL)).toEqual(entries(FULL));
  });

  it('parses every key into its state field', () => {
    const s = parseWorkspaceUrl(new URLSearchParams(FULL));
    expect(s).toMatchObject({
      view: 'table',
      queue: 'to_label',
      scope: { kind: 'cluster', id: 'abc' },
      focus: { kind: 'item', id: 'crop1' },
      rows: 'images',
      columns: ['class', 'conf'],
      thumb: 'l',
      q: 'red widget',
      similarTo: 'crop9',
      map: { zoom: 12, lat: 51.51, lon: -0.13 },
    });
    expect(s.sort).toEqual({
      gallery: 'taken_desc',
      clusters: 'size_desc',
      detections: 'conf_asc',
      review: 'rank',
      table: 'class',
    });
  });

  it('ignores an unknown view and keeps the rest', () => {
    const s = parseWorkspaceUrl(new URLSearchParams('view=nope&queue=to_label'));
    expect(s.view).toBeNull();
    expect(s.queue).toBe('to_label');
  });

  it('drops a malformed scope, focus and map', () => {
    const s = parseWorkspaceUrl(
      new URLSearchParams('scope=bogus%3A1&focus=item&map=1%2F2&rows=x&thumb=xl'),
    );
    expect(s).toMatchObject({
      scope: null,
      focus: null,
      map: null,
      rows: 'detections',
      thumb: 'm',
    });
  });

  it('keeps a colon inside an id', () => {
    const s = parseWorkspaceUrl(new URLSearchParams('scope=image%3Aa%3Ab'));
    expect(s.scope).toEqual({ kind: 'image', id: 'a:b' });
  });

  it('leaves keys it was not given and deletes keys set to null', () => {
    const p = new URLSearchParams('class_name=widget&view=map&queue=q1');
    writeWorkspaceUrl(p, { view: 'clusters', queue: null });
    expect(p.toString()).toBe('class_name=widget&view=clusters');
  });

  it('never writes the defaults', () => {
    const p = new URLSearchParams('rows=images&thumb=l');
    writeWorkspaceUrl(p, { rows: 'detections', thumb: 'm' });
    expect(p.toString()).toBe('');
  });

  it('owns no item-filter key', () => {
    expect(WORKSPACE_URL_KEYS).not.toContain('class_name');
    expect(WORKSPACE_URL_KEYS).toContain('sort_clusters');
  });
});

describe('clusterScopeSearch', () => {
  it('opens a cluster in the Clusters view', () => {
    expect(clusterScopeSearch('u 1')).toBe('?view=clusters&scope=cluster%3Au+1');
  });
});
