/**
 * The cohort preview's cards refresh the preview after an image Reprocess
 * (W10): both card kinds pass `onreprocessed`, which re-runs the cohort
 * preview query. A source scan: mounting the whole train page to drive a
 * reprocess dialog from a cohort card is out of proportion, and the
 * card-side behavior is covered by CropCard.test.ts / SlotCard.test.ts.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { normalize } from '$lib/testing/sourceScan';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = normalize(readFileSync(path.resolve(here, './+page.svelte'), 'utf-8'));

describe('train cohort preview cards', () => {
  it('re-run the preview when a slot or crop card reprocesses its image', () => {
    const wired = src.match(
      /onreprocessed=\{\(\) => void loadCohortPreview\(group, activeCohort\)\}/g,
    );
    expect(wired).toHaveLength(2);
    expect(src).toMatch(/<SlotCard .{0,300}?onreprocessed=/);
    expect(src).toMatch(/<CropCard .{0,300}?onreprocessed=/);
  });
});
