/**
 * W3 (docs/design/logic-moves-adoption-plan-2026-09-24.md) — the
 * dashboard's "Run VLM on cluster" modal uses POST
 * /vlm/label_cluster/{id} + status polling instead of the old
 * fetch-crops-then-chunk-of-64 loop against /vlm/label_batch. Static
 * source scan (no component-mount harness in this repo — see
 * clusters/[id]/clusterMoveRace.test.ts's doc comment).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('W3: dashboard runVlm uses POST /vlm/label_cluster/{id} + status polling', () => {
  it('imports pollAutoLabelJob alongside runVlmOnCluster', () => {
    expect(src).toMatch(/pollAutoLabelJob/);
    expect(src).toMatch(/runVlmOnCluster/);
  });

  it('runVlm calls runVlmOnCluster then polls via pollAutoLabelJob', () => {
    const fn = src.match(/async function runVlm\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/await runVlmOnCluster\(id\)/);
    expect(fn).toMatch(/await pollAutoLabelJob\(/);
    expect(fn).not.toMatch(/label_batch/);
  });

  it('renders the polled job stage inline in the modal', () => {
    expect(src).toMatch(/vlmJob\.stage/);
  });
});
