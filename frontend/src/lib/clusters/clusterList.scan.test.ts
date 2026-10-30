/**
 * The cluster list renders the order the server served. A client sort, the
 * old client-side "unlabeled only" narrowing and per-card thumbnail requests
 * are gone; a re-ranking added back fails here.
 */
import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const LIB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const FILES = [
  'clusters/clusterListController.svelte.ts',
  'clusters/clusterListFilter.ts',
  'clusters/persistedFilter.ts',
  'clusters/clusterCardText.ts',
  'components/clusters/ClustersGrid.svelte',
  'components/clusters/ClusterGridCard.svelte',
  'components/clusters/ClustersToolbar.svelte',
  'components/clusters/ClusterListFilterBar.svelte',
  'components/clusters/ClustersStatusBar.svelte',
];

const RERANK = /\.(sort|toSorted|reverse|toReversed)\(|sortClusters|unlabeledOnly/;

describe('cluster list is rendered in served order', () => {
  it('the rerank pattern catches the spellings (negative control)', () => {
    expect(RERANK.test('out.sort((a, b) => a.size - b.size)')).toBe(true);
    expect(RERANK.test('list.toSorted(cmp)')).toBe(true);
    expect(RERANK.test('sortClusters(list, mode)')).toBe(true);
    expect(RERANK.test('clusterPager.items.slice(0, shown)')).toBe(false);
  });

  it.each(FILES)('%s does not sort, reverse or narrow the cards', (file) => {
    const full = path.join(LIB, file);
    expect(existsSync(full), `${file} exists`).toBe(true);
    const lines = readFileSync(full, 'utf-8').split('\n');
    const hits = lines.filter((l) => !l.trim().startsWith('//') && RERANK.test(l));
    expect(hits).toEqual([]);
  });

  it('the per-card representative fetch is deleted', () => {
    expect(existsSync(path.join(LIB, 'clusters/displayOrderRepresentatives.ts'))).toBe(
      false,
    );
    const src = readFileSync(
      path.join(LIB, 'clusters/clusterListController.svelte.ts'),
      'utf-8',
    );
    expect(src).not.toMatch(/cluster_id:\s*id/);
    expect(src).not.toMatch(/idsNeedingRepresentatives|loadMoreRepresentatives/);
  });
});
