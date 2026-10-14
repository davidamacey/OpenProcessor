/**
 * A cluster card's class link uses the served `dominant_class_id`, and a
 * name lookup (deep link, slot inventory card) resolves to the ACTIVE class
 * when a deprecated class carries the same name. Static source scan, like
 * the sibling /clusters page tests.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const LIB = path.resolve(here, '../../../../lib');
const pageSrc = readFileSync(path.join(here, '+page.svelte'), 'utf-8');
const cardSrc = readFileSync(
  path.join(LIB, 'components/clusters/ClusterGridCard.svelte'),
  'utf-8',
);
const inventorySrc = readFileSync(
  path.join(LIB, 'clusters/slotInventory.svelte.ts'),
  'utf-8',
);
const src = [
  pageSrc,
  cardSrc,
  inventorySrc,
  readFileSync(path.join(LIB, 'clusters/clusterListController.svelte.ts'), 'utf-8'),
  readFileSync(path.join(LIB, 'clusters/clusterSearchModes.svelte.ts'), 'utf-8'),
  readFileSync(path.join(LIB, 'components/clusters/ClustersGrid.svelte'), 'utf-8'),
  readFileSync(path.join(LIB, 'components/clusters/ClustersToolbar.svelte'), 'utf-8'),
].join('\n');

describe('/clusters class lookups', () => {
  it('a slot-bound card click routes by the served dominant_class_id, not a name lookup', () => {
    const fn = cardSrc.slice(cardSrc.indexOf('P2.11 fix'), cardSrc.indexOf('</script>'));
    expect(fn).toMatch(/c\.dominant_class_id/);
    expect(fn).not.toMatch(/toLowerCase/);
  });

  it('every name-to-class lookup goes through the active-first normalized lookup', () => {
    expect(pageSrc).toMatch(/findActiveClassByName\(classesStore\.classes, v\)/);
    expect(inventorySrc).toMatch(
      /findActiveClassByName\(classesStore\.classes, className\)/,
    );
    expect(src).not.toMatch(/c\.name\.toLowerCase\(\)/);
  });
});
