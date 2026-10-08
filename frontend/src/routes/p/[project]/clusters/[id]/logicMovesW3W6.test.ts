/**
 * W3 + W6 (docs/design/logic-moves-adoption-plan-2026-09-24.md) —
 * /clusters/[id]'s VLM-on-cluster run and the served cluster shape. No
 * component-mount harness in this repo (see clusterMoveRace.test.ts's doc
 * comment), so this is a static source scan, same convention as
 * logicMovesW1.test.ts.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('W3: runVlm uses POST /vlm/label_cluster/{id} + status polling, not the old batch loop', () => {
  it('imports pollAutoLabelJob alongside runVlmOnCluster', () => {
    expect(src).toMatch(/pollAutoLabelJob/);
    expect(src).toMatch(/runVlmOnCluster/);
  });

  it('runVlm calls runVlmOnCluster then polls via pollAutoLabelJob, never fetching a crop page itself', () => {
    const fn = src.match(/async function runVlm\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/await runVlmOnCluster\(clusterId,/);
    expect(fn).toMatch(/await pollAutoLabelJob\(/);
    // The old implementation fetched {API_PREFIX}/crops itself before
    // chunking into {API_PREFIX}/vlm/label_batch — neither should survive.
    expect(fn).not.toMatch(/getCrops\(/);
    expect(fn).not.toMatch(/label_batch/);
  });

  it('renders the job stage/progress from the polled state, not a client-computed percentage', () => {
    expect(src).toMatch(/vlmJob\.stage/);
    expect(src).toMatch(/vlmJob\.processed/);
    expect(src).toMatch(/vlmJob\.total/);
  });
});

describe('W6: class-for-cluster lookup uses the served cluster_kind, not an id-equality guess', () => {
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

describe('W6: the core cut line uses the served cluster_is_core, not a 0.75 constant', () => {
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
