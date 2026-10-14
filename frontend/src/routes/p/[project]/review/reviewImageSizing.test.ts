/**
 * The /review crop/region image never collapses to 0px when Details opens
 * (the metadata region scrolls independently), and the crop thumbnail
 * upscales to fill its panel at any viewport. Static source scan: the layout
 * classes under test are not renderable under jsdom plus mount, and "did the
 * source really change" is exactly the pin a mount test can't beat (see
 * slotReviewCharacterization.test.ts).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('the crop/region image never collapses to 0px when Details opens', () => {
  it('the image wrapper is shrink-0 with a floor min-height, not flex-1', () => {
    // DQ-M5 (2026-09-24 data-quality pass) added a `max-h-[46%]` ceiling
    // alongside the floor — the wrapper is still shrink-0 with min-height,
    // just no longer unbounded above.
    // F8 D4 (2026-09-25): a fixed floor height below lg (the body scrolls
    // there), the floor + ceiling from lg up; overflow-hidden so a region
    // canvas can't paint over the first metadata row.
    expect(src).toMatch(
      /flex h-\[\d+px\] shrink-0 items-center justify-center overflow-hidden bg-zinc-950 lg:h-auto lg:max-h-\[\d+%\] lg:min-h-\[\d+px\]/,
    );
  });

  it('the metadata/Details region below the image scrolls independently', () => {
    // From lg up it scrolls in its own region; below lg the whole review
    // body scrolls instead (F8 D4: the inner pane was ~79px at 800px).
    expect(src).toMatch(/class="mt-3 pr-1 lg:min-h-0 lg:flex-1 lg:overflow-y-auto"/);
  });
});

describe('the /review crop thumbnail upscales to fill its panel, at any viewport', () => {
  it('the plain-<img> crop branch uses h-full w-full, not max-h-full max-w-full (which never upscales)', () => {
    const block = src.match(/\{:else\}\s*\n\s*<!-- p9[\s\S]*?<img[\s\S]*?\/>/)?.[0];
    expect(block).not.toBeUndefined();
    expect(block).toMatch(/class="h-full w-full object-contain"/);
    expect(block).not.toMatch(/class="max-h-full max-w-full object-contain"/);
  });
});
