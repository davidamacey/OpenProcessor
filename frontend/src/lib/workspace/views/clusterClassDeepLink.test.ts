/**
 * A `class_name` deep link in the Workspace Clusters view resolves against the loaded
 * class registry. Static source scan (no component-mount harness; see
 * clusters/[id]/clusterMoveRace.test.ts's doc comment).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, 'ClusterListView.svelte'), 'utf-8');

describe('a `class_name` deep link resolves against the loaded registry', () => {
  it('classFilter resolves the single filter class through the normalized active-class lookup', () => {
    const fn = src.match(/const classFilter = \$derived\([\s\S]*?\n {2}\);/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/findActiveClassByName\(classesStore\.classes, classFilterName\)/);
  });
});
