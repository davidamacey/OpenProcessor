/**
 * class-id-display-audit-2026-09-26 LOW finding: `clusterName` used to
 * resolve `dominant_class_id` against the registry (`classesStore`) before
 * falling back to the served `dominant_class_name`. Both are safe by
 * construction for a real class cluster (`cluster_kind === 'class'` implies
 * `cluster_id === class_id`), but for consistency with the thin-frontend
 * rule the served name should be preferred. `clsForCluster` (the registry
 * object) is kept only for what genuinely needs a live registry row —
 * `liveClassId` (hotkey/assign) and the validated/count figures — not for
 * display naming.
 *
 * Same static source-scan convention as the other clusters/[id] logic-
 * moves tests (dqM3CutLineVisibility.test.ts) — no mount harness for a
 * page this size.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('class-id-display-audit-2026-09-26: clusterName prefers the served name', () => {
  it('clusterName reads cluster.dominant_class_name before the registry lookup', () => {
    expect(src).toMatch(
      /const clusterName = \$derived\(\s*cluster\?\.dominant_class_name \?\? clsForCluster\?\.name \?\? null,?\s*\);/,
    );
  });

  it('clsForCluster (the registry row) is still used for the live count/validated figures', () => {
    expect(src).toMatch(
      /const liveClassId = \$derived\(clsForCluster\?\.id \?\? null\);/,
    );
    expect(src).toMatch(/clsForCluster\.validated_count/);
    expect(src).toMatch(/clsForCluster\.count/);
  });
});
