/**
 * Coordinator minor (2026-09-25): on the region tab a MISTAKENNESS score
 * chip rendered inside the Detector row. It belongs only in the Scores
 * row. Source scan: the route page isn't mountable in this harness.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('/review score chip placement', () => {
  it('the mistakenness chip renders once, in the Scores row', () => {
    expect(src.match(/label="mistakenness"/g)).toHaveLength(1);
    const scores = src.indexOf('<dt class="text-zinc-500">Scores</dt>');
    const chip = src.indexOf('label="mistakenness"');
    const detectorRow = src.indexOf('<span class="text-zinc-500">Detector</span>');
    expect(scores).toBeGreaterThan(-1);
    expect(chip).toBeGreaterThan(scores);
    expect(chip).toBeLessThan(detectorRow);
  });
});
