/**
 * class-id-display-audit-2026-09-26 LOW finding: `clusterName` used to
 * resolve `dominant_class_id` against the registry (`classesStore`) before
 * falling back to the served `dominant_class_name`. Both are safe by
 * construction, but for consistency with the thin-frontend rule the served
 * name should be preferred. `clsForCluster` (the registry
 * object) is kept only for what genuinely needs a live registry row —
 * `liveClassId` (hotkey/assign) and the validated/count figures — not for
 * display naming.
 *
 * Same static source-scan convention as the other clusters/[id]
 * tests (clusterDetailServedFields.test.ts) — no mount harness for a
 * page this size.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const LIB = path.resolve(here, '../..');
const src = readFileSync(
  path.join(LIB, 'clusters/clusterDetailController.svelte.ts'),
  'utf-8',
);
const liveSrc = readFileSync(
  path.join(LIB, 'clusters/clusterDetailLive.svelte.ts'),
  'utf-8',
);
const detailsSrc = readFileSync(
  path.join(LIB, 'workspace/details/ClusterDetails.svelte'),
  'utf-8',
);

describe('class-id-display-audit-2026-09-26: clusterName prefers the served name', () => {
  it('clusterName reads cluster.dominant_class_name before the registry lookup', () => {
    expect(src).toMatch(
      /const clusterName = \$derived\(\s*cluster\?\.dominant_class_name \?\? clsForCluster\?\.name \?\? null,?\s*\);/,
    );
  });

  it('clsForCluster (the registry row) is still used for the live count/validated figures', () => {
    expect(liveSrc).toMatch(
      /const liveClassId = \$derived\(view\.clsForCluster\?\.id \?\? null\);/,
    );
    expect(detailsSrc).toMatch(/cls\.validated_count/);
    expect(detailsSrc).toMatch(/cls\.count/);
  });
});
