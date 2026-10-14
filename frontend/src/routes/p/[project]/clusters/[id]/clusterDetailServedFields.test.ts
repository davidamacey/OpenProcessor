/**
 * /clusters/[id] reads the served cluster shape instead of guessing: the
 * class for a cluster comes from `cluster_kind`/`dominant_class_id`, and the
 * core cut line from `cluster_is_core`. Static source scan (no
 * component-mount harness; see clusterMoveRace.test.ts's doc comment).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('class-for-cluster lookup uses the served cluster_kind, not an id-equality guess', () => {
  it('clsForCluster reads cluster.cluster_kind and cluster.dominant_class_id', () => {
    const clsBlock = src.match(/const clsForCluster = \$derived\(([\s\S]*?)\);/)?.[0];
    expect(clsBlock).not.toBeUndefined();
    expect(clsBlock).toMatch(/cluster\?\.cluster_kind === 'class'/);
    expect(clsBlock).toMatch(/cluster\.dominant_class_id/);
    // The old implementation assumed cluster_id === class_id and matched
    // classesStore entries against clusterId directly.
    expect(clsBlock).not.toMatch(/c\.id === clusterId/);
  });
});

describe('the core cut line uses the served cluster_is_core, not a 0.75 constant', () => {
  it('cutLine is computed from filteredCrops via computeCutLine, which reads cluster_is_core', () => {
    expect(src).toMatch(
      /const cutLine = \$derived\.by\(\(\) =>\s*computeCutLine\(filteredCrops, orderMode === 'default'\),?\s*\);/,
    );
    expect(src).not.toMatch(/similarity_to_centroid/);
  });

  it('imports computeCutLine from the shared lib rather than reimplementing the boundary scan inline (one place for the core-first cut)', () => {
    expect(src).toMatch(/import \{ computeCutLine \} from '\$lib\/clusters\/cutLine';/);
  });
});
