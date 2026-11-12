/**
 * /clusters/[id] reads the served cluster shape instead of guessing: the
 * class for a cluster comes from its `dominant_class_name`, and the retired
 * geometry (core cut line, sub-clusters) is gone. Static source scan (no
 * component-mount harness; see clusterMoveRace.test.ts's doc comment).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const LIB = path.resolve(here, '../../../../../lib');
const src = readFileSync(
  path.join(LIB, 'clusters/clusterDetailController.svelte.ts'),
  'utf-8',
);

describe('class-for-cluster lookup reads the served class name, not an id guess', () => {
  it('clsForCluster matches the registry class by the card’s dominant_class_name', () => {
    const clsBlock = src.match(/const clsForCluster = \$derived\(([\s\S]*?)\);/)?.[0];
    expect(clsBlock).not.toBeUndefined();
    expect(clsBlock).toMatch(/cluster\?\.dominant_class_name/);
    expect(clsBlock).toMatch(/c\.name === cluster!\.dominant_class_name/);
    // Class identity is the name: no registry row is matched by an id.
    expect(clsBlock).not.toMatch(/dominant_class_id|cluster_kind|c\.id ===/);
  });
});

describe('the detail view carries no retired cluster geometry', () => {
  it('has no core cut line, sub-cluster tabs or grouping (cluster_is_core, subid are not served)', () => {
    expect(src).not.toMatch(/cutLine|computeCutLine|cluster_is_core/);
    expect(src).not.toMatch(/subTab|subClusterIds|groupBySubcluster|cluster_subid/);
    expect(src).not.toMatch(/similarity_to_centroid/);
  });

  it('asks for the best-scored-first order the server serves (core_first), keyed by the cluster uid', () => {
    expect(src).toMatch(/'core_first'/);
    expect(src).toMatch(/getCluster\(clusterUid, cursor, pageSize/);
    expect(src).not.toMatch(/clusterId/);
  });
});
