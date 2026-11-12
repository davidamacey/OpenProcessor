/**
 * Coordinator minor (2026-09-25): on the region tab a MISTAKENNESS score
 * chip rendered inside the Detector row. It belongs only in the Scores
 * row. Source scan: the route page isn't mountable in this harness.
 */
import { describe, expect, it } from 'vitest';
import { reviewPageSource, reviewSource } from '$lib/testing/reviewPageSources';

const metaSrc = reviewSource('lib/components/review/ReviewItemMeta.svelte');
const panelSrc = reviewSource('lib/components/review/ReviewSlotPanel.svelte');
const paneSrc = reviewSource('lib/components/review/ReviewCropPane.svelte');

describe('/review score chip placement', () => {
  it('the mistakenness chip renders once, in the Scores row', () => {
    expect(reviewPageSource().match(/label="mistakenness"/g)).toHaveLength(1);
    const scores = metaSrc.indexOf('<dt class="text-zinc-500">Scores</dt>');
    const chip = metaSrc.indexOf('label="mistakenness"');
    expect(scores).toBeGreaterThan(-1);
    expect(chip).toBeGreaterThan(scores);
    // The Detector row lives in the slot panel, which the crop pane
    // renders after the item rows: the chip stays above it, never beside it.
    expect(panelSrc).toContain('<span class="text-zinc-500">Detector</span>');
    expect(paneSrc.indexOf('<ReviewItemMeta')).toBeGreaterThan(-1);
    expect(paneSrc.indexOf('<ReviewItemMeta')).toBeLessThan(
      paneSrc.indexOf('<ReviewSlotPanel'),
    );
  });
});
