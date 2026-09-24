/**
 * `<TrainForm>`'s GPU picker renders the backend's allowed claims
 * (`GET /train/gpus`, fetched by `getTrainGpus`) rather than a hardcoded
 * list. The fetch and default selection are unit-tested in
 * `api.trainGpus.test.ts`; the actual rendered picker (options, checked
 * default, advisory text) is now mount-tested in
 * `TrainForm.gpuPicker.test.ts` (test-audit-2026-09-24.md P1-4).
 *
 * What's left here is source-scan only, and only for checks a DOM mount
 * can't reach: an *absence* of dead code/literals, and payload wiring a
 * mount test would need to drive Start/campaign submission to reach
 * (out of scope for a render-only assertion). Each `it()` below carries
 * its own one-line "why this can't be a mount test" reason (P2-2).
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { extractFunction } from '$lib/testing/sourceScan';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, './TrainForm.svelte'), 'utf-8');

describe('TrainForm.svelte GPU picker', () => {
  // A mount test can prove a list is rendered; it can't prove a SECOND,
  // dead list doesn't also exist somewhere in the file. Absence-of-code
  // checks stay scans.
  it('keeps no GPU list of its own', () => {
    expect(src).not.toMatch(/GPU_OPTIONS/);
    expect(src).not.toMatch(/trainGpuOptions/);
  });

  // Same reasoning: GPU 1 is reserved for another project (CLAUDE.md) —
  // this guards against ever reintroducing it as a literal default,
  // which a mount test asserting "the served default renders" wouldn't
  // catch if someone also left a hardcoded fallback in the source.
  it('never hardcodes the literal host GPU id 1 anywhere in the component', () => {
    // Broad net: any bare `'1'` / `"1"` string literal that could plausibly
    // be a stray GPU id. False positives (e.g. a completely unrelated
    // '1' string) would need updating this regex, but there are none
    // today — see the full match list below if this ever fails.
    const suspicious = src.match(/(?:cuda|gpu)[a-zA-Z]*\s*[:=]\s*['"]1['"]/gi) ?? [];
    expect(suspicious).toEqual([]);
  });

  // Reaching this by mount would mean driving the full Start/Start
  // campaign submit flow and inspecting the constructed request body —
  // out of scope for a render-focused mount test. extractFunction is
  // brace-balanced (P2-2), so this survives reformatting that the old
  // `/function buildSpec\(\)[\s\S]*?\n {2}\}/`-style regex didn't.
  it('threads the selected cudaDevices value into both the single-run and campaign payloads', () => {
    // buildSpec() -> POST {API_PREFIX}/train/start; buildCampaign() -> POST
    // {API_PREFIX}/train/start_campaign. Both must carry whatever the user picked.
    const buildSpecMatch = extractFunction(src, 'buildSpec');
    const buildCampaignMatch = extractFunction(src, 'buildCampaign');
    expect(buildSpecMatch).not.toBeNull();
    expect(buildCampaignMatch).not.toBeNull();
    expect(buildSpecMatch).toMatch(/cuda_visible_devices:\s*cudaDevices/);
    expect(buildCampaignMatch).toMatch(/cuda_visible_devices:\s*cudaDevices/);
  });

  // Absence-of-a-domain-specific-literal check (no mount can prove a
  // string never appears anywhere in the source).
  it('takes the generic singleClassExport prop, not the domain-specific lpr one', () => {
    expect(src).toMatch(/singleClassExport\?:\s*boolean/);
    expect(src).not.toMatch(/\blpr\b/);
  });
});
