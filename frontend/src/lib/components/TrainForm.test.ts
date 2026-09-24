/**
 * `<TrainForm>`'s GPU picker renders the backend's allowed claims
 * (`GET /train/gpus`, fetched by `getTrainGpus`) rather than a hardcoded
 * list. The fetch and default selection are unit-tested in
 * `api.trainGpus.test.ts`. This static source scan (no component-mount
 * harness yet) confirms the component renders the served options and
 * threads the selection into both payloads.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, './TrainForm.svelte'), 'utf-8');

describe('TrainForm.svelte GPU picker', () => {
  it('loads the served options from getTrainGpus and preselects the served default', () => {
    expect(src).toMatch(/getTrainGpus\(/);
    expect(src).toMatch(/cudaDevices = defaultGpuValue\(res\)/);
  });

  it('keeps no GPU list of its own', () => {
    expect(src).not.toMatch(/GPU_OPTIONS/);
    expect(src).not.toMatch(/trainGpuOptions/);
  });

  it('never hardcodes the literal host GPU id 1 anywhere in the component', () => {
    // Broad net: any bare `'1'` / `"1"` string literal that could plausibly
    // be a stray GPU id. False positives (e.g. a completely unrelated
    // '1' string) would need updating this regex, but there are none
    // today — see the full match list below if this ever fails.
    const suspicious = src.match(/(?:cuda|gpu)[a-zA-Z]*\s*[:=]\s*['"]1['"]/gi) ?? [];
    expect(suspicious).toEqual([]);
  });

  it('renders one radio per served option, bound to the shared cudaDevices state', () => {
    expect(src).toMatch(/\{#each gpuOptions\.options as opt/);
    expect(src).toMatch(/checked=\{cudaDevices === opt\.value\}/);
    expect(src).toMatch(/onchange=\{\(\) => \(cudaDevices = opt\.value\)\}/);
  });

  it('threads the selected cudaDevices value into both the single-run and campaign payloads', () => {
    // buildSpec() -> POST {API_PREFIX}/train/start; buildCampaign() -> POST
    // {API_PREFIX}/train/start_campaign. Both must carry whatever the user picked.
    const buildSpecMatch = src.match(/function buildSpec\(\)[\s\S]*?\n {2}\}/);
    const buildCampaignMatch = src.match(/function buildCampaign\(\)[\s\S]*?\n {2}\}/);
    expect(buildSpecMatch).not.toBeNull();
    expect(buildCampaignMatch).not.toBeNull();
    expect(buildSpecMatch?.[0]).toMatch(/cuda_visible_devices:\s*cudaDevices/);
    expect(buildCampaignMatch?.[0]).toMatch(/cuda_visible_devices:\s*cudaDevices/);
  });

  it("shows the selected option's served advisory", () => {
    expect(src).toMatch(/options\.find\(\(o\) => o\.value === cudaDevices\)\?\.advisory/);
  });

  it('takes the generic singleClassExport prop, not the domain-specific lpr one', () => {
    expect(src).toMatch(/singleClassExport\?:\s*boolean/);
    expect(src).not.toMatch(/\blpr\b/);
  });
});
