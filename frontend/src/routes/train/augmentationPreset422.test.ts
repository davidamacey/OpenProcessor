/**
 * OpenProcessor df01309: `POST {API_PREFIX}/train/start` and
 * `/train/start_campaign` 422 on an unknown `augmentation.preset` with
 * `{detail: {message, field: 'augmentation.preset', valid_presets}}`.
 * `maybeRenderPreflight` (`/train/+page.svelte`) already read
 * `body.detail.message` generically before this change — this proves it
 * also surfaces the served `valid_presets` list, not just the bare
 * message, and does so without special-casing every other 422 shape
 * (`body.detail.field` absent ⇒ unchanged behavior).
 *
 * Same static source-scan convention as `interactivePassFixes.test.ts`/
 * `logicMovesW6TrainingCohorts.test.ts` — no component-mount harness for
 * this page (many polling intervals/onMount effects; a route-level
 * mount would leak timers across tests).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, './+page.svelte'), 'utf-8');

function extractFunction(name: string): string {
  const start = src.indexOf(`function ${name}(`);
  expect(start).toBeGreaterThan(-1);
  let depth = 0;
  let i = src.indexOf('{', start);
  const bodyStart = i;
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}') {
      depth--;
      if (depth === 0) break;
    }
  }
  return src.slice(bodyStart, i + 1);
}

describe('maybeRenderPreflight surfaces the augmentation-preset 422 detail', () => {
  const body = extractFunction('maybeRenderPreflight');

  it('reads the served field/valid_presets off body.detail', () => {
    expect(body).toMatch(/body\.detail\?\.field/);
    expect(body).toMatch(/body\.detail\.valid_presets/);
  });

  it("only appends valid_presets when field === 'augmentation.preset'", () => {
    expect(body).toMatch(/field\s*===\s*'augmentation\.preset'/);
  });

  it('still falls back to the bare message when no field/valid_presets are present', () => {
    expect(body).toMatch(/body\.detail\?\.message\s*\?\?\s*err\.message/);
  });
});
