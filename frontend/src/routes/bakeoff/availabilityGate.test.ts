/**
 * Static source scan for /bakeoff's availability gate
 * (docs/design/bakeoff-train-genericization-plan-2026-09-21.md §2.5/§6
 * commit 1). Same convention as `../train/datasetExportGate.test.ts` —
 * this repo has no `@testing-library/svelte` harness, so this asserts the
 * wiring a reviewer would check by eye.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const pageSrc = readFileSync(path.resolve(here, './+page.svelte'), 'utf-8');
const layoutSrc = readFileSync(path.resolve(here, '../+layout.svelte'), 'utf-8');

describe('/bakeoff availability gate', () => {
  it('+page.svelte imports the availability store', () => {
    expect(pageSrc).toMatch(
      /import\s*\{[^}]*bakeoffAvailability[^}]*\}\s*from\s*['"]\$lib\/bakeoffAvailability\.svelte['"]/,
    );
  });

  // onMount must gate the discovery GETs (controller.init) behind the
  // probe — firing them unconditionally is the exact "never 404" violation
  // this gate exists to remove (see /train's datasetExportGate.test.ts
  // sibling assertion).
  it('onMount does not call controller.init before the availability check', () => {
    const onMountMatch = pageSrc.match(/onMount\(async \(\) => \{[\s\S]*?\n {2}\}\);/);
    expect(onMountMatch).not.toBeNull();
    const body = onMountMatch![0];
    const availabilityIdx = body.indexOf('bakeoffAvailability.available === false');
    const refreshIdx = body.indexOf('controller.init()');
    expect(availabilityIdx).toBeGreaterThan(-1);
    expect(refreshIdx).toBeGreaterThan(-1);
    expect(availabilityIdx).toBeLessThan(refreshIdx);
  });

  it('+layout.svelte wraps the /bakeoff anchor in a conditional referencing bakeoffAvailability', () => {
    const anchorMatch = layoutSrc.match(
      /\{#if[^}]*bakeoffAvailability[^}]*\}\s*<a\s+href="\/bakeoff"/,
    );
    expect(anchorMatch).not.toBeNull();
  });
});
