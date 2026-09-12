/**
 * `<TrainForm>`'s GPU picker (follow-up gap 1,
 * docs/design/audit-remediation-plan-2026-09.md Appendix D item 2,
 * 2026-09-11). This repo has no `@testing-library/svelte` and adding one
 * is out of scope for this follow-up (see `StrategyBar.test.ts`'s note),
 * so — same convention — this is a static source scan rather than a
 * mounted-component test: it asserts the wiring a reviewer would check
 * by eye, exhaustively, for both the exhaustive-options question and the
 * payload-threading question.
 *
 * The exhaustive "never offers GPU 1" claim itself is unit-tested
 * directly (not by scanning source text) in `trainGpuOptions.test.ts`,
 * against the real `GPU_OPTIONS` array — this file only has to confirm
 * `TrainForm.svelte` actually uses that shared array rather than its own
 * (possibly drifted) inline copy.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, './TrainForm.svelte'), 'utf-8');

describe('TrainForm.svelte GPU picker', () => {
  it('imports GPU_OPTIONS from the shared, unit-tested module instead of an inline array', () => {
    expect(src).toMatch(/import\s*\{[^}]*GPU_OPTIONS[^}]*\}\s*from\s*['"]\$lib\/trainGpuOptions['"]/);
  });

  it('does not redeclare its own inline GPU_OPTIONS constant (single source of truth)', () => {
    expect(src).not.toMatch(/const\s+GPU_OPTIONS\s*=/);
  });

  it('never hardcodes the literal host GPU id 1 anywhere in the component', () => {
    // Broad net: any bare `'1'` / `"1"` string literal that could plausibly
    // be a stray GPU id. False positives (e.g. a completely unrelated
    // '1' string) would need updating this regex, but there are none
    // today — see the full match list below if this ever fails.
    const suspicious = src.match(/(?:cuda|gpu)[a-zA-Z]*\s*[:=]\s*['"]1['"]/gi) ?? [];
    expect(suspicious).toEqual([]);
  });

  it('renders one radio per GPU_OPTIONS entry, bound to the shared cudaDevices state', () => {
    expect(src).toMatch(/\{#each GPU_OPTIONS as opt/);
    expect(src).toMatch(/checked=\{cudaDevices === opt\.value\}/);
    expect(src).toMatch(/onchange=\{\(\) => \(cudaDevices = opt\.value\)\}/);
  });

  it('threads the selected cudaDevices value into both the single-run and campaign payloads', () => {
    // buildSpec() -> POST /curation/train/start; buildCampaign() -> POST
    // /curation/train/start_campaign. Both must carry whatever the user picked.
    const buildSpecMatch = src.match(/function buildSpec\(\)[\s\S]*?\n {2}\}/);
    const buildCampaignMatch = src.match(/function buildCampaign\(\)[\s\S]*?\n {2}\}/);
    expect(buildSpecMatch).not.toBeNull();
    expect(buildCampaignMatch).not.toBeNull();
    expect(buildSpecMatch?.[0]).toMatch(/cuda_visible_devices:\s*cudaDevices/);
    expect(buildCampaignMatch?.[0]).toMatch(/cuda_visible_devices:\s*cudaDevices/);
  });

  it('renders the shared gpuAdvisory(...) text instead of a one-off inline conditional', () => {
    expect(src).toMatch(/gpuAdvisory\(cudaDevices\)/);
  });
});
