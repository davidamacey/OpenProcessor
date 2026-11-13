/**
 * A cluster card's class link carries the served `dominant_class_name`, and a
 * name lookup (deep link, slot inventory card) resolves to the ACTIVE class
 * when a deprecated class carries the same name. Static source scan, like
 * the sibling /clusters page tests.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const LIB = path.resolve(here, '../..');
const pageSrc = readFileSync(path.join(here, 'ClusterListView.svelte'), 'utf-8');
const cardSrc = readFileSync(path.join(LIB, 'clusters/clusterHref.ts'), 'utf-8');
const src = [
  pageSrc,
  cardSrc,
  readFileSync(path.join(LIB, 'clusters/clusterListController.svelte.ts'), 'utf-8'),
  readFileSync(path.join(LIB, 'clusters/clusterSearchModes.svelte.ts'), 'utf-8'),
  readFileSync(path.join(LIB, 'components/clusters/ClustersGrid.svelte'), 'utf-8'),
  readFileSync(path.join(LIB, 'components/clusters/ClusterListHeader.svelte'), 'utf-8'),
].join('\n');

describe('/clusters class lookups', () => {
  it('a slot-bound card click routes by the served dominant_class_name (class identity is the name)', () => {
    const fn = cardSrc.slice(cardSrc.indexOf('export function clusterHref'));
    expect(fn).toMatch(/encodeURIComponent\(dominantClassName\)/);
    expect(fn).not.toMatch(/dominant_class_id/);
    expect(fn).not.toMatch(/toLowerCase/);
  });

  it('every name-to-class lookup goes through the active-first normalized lookup', () => {
    expect(pageSrc).toMatch(
      /findActiveClassByName\(classesStore\.classes, classFilterName\)/,
    );
    expect(src).not.toMatch(/c\.name\.toLowerCase\(\)/);
  });
});
