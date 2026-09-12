/**
 * `GPU_OPTIONS` is the exhaustive allowlist for the `/train` form's GPU
 * picker (follow-up gap 1, docs/design/audit-remediation-plan-2026-09.md
 * Appendix D item 2, 2026-09-11). This host has 3 GPUs — slot 0 and slot
 * 2 are A6000s the trainer may use; slot 1 is a 3080 Ti dedicated to an
 * unrelated app and must never be offered as a training target.
 *
 * Before this module existed, the picker was an inline array in
 * `TrainForm.svelte` with only 2 entries ('0,2' and '0') — there was no
 * way to select GPU 2 alone at all, let alone verify GPU 1 could never
 * sneak in. Extracted here (mirroring `strategies.ts`'s pattern) so it's
 * unit-testable without a component-mount harness — this repo has no
 * `@testing-library/svelte` (see `StrategyBar.test.ts`).
 */

import { describe, expect, it } from 'vitest';
import { FORBIDDEN_GPU_ID, GPU_OPTIONS, gpuAdvisory } from './trainGpuOptions';

describe('GPU_OPTIONS', () => {
  it('never offers GPU 1 (3080 Ti) as a selectable value, in any option', () => {
    for (const opt of GPU_OPTIONS) {
      const hostIds = opt.value.split(',').map((s) => s.trim());
      expect(hostIds).not.toContain(FORBIDDEN_GPU_ID);
    }
  });

  it('exposes exactly the 3 valid host-GPU claims: both, GPU 0 alone, GPU 2 alone', () => {
    const values = GPU_OPTIONS.map((o) => o.value).sort();
    expect(values).toEqual(['0', '0,2', '2']);
  });

  it('offers a single-GPU-2 option — the gap this follow-up closes', () => {
    const gpu2Only = GPU_OPTIONS.find((o) => o.value === '2');
    expect(gpu2Only).toBeDefined();
    expect(gpu2Only?.warn).toBe(true);
  });

  it('every option value only ever references GPU 0 and/or GPU 2', () => {
    for (const opt of GPU_OPTIONS) {
      const hostIds = opt.value.split(',').map((s) => s.trim());
      for (const id of hostIds) {
        expect(['0', '2']).toContain(id);
      }
    }
  });
});

describe('gpuAdvisory', () => {
  it('warns single-GPU-0 that Gemma worker stays alive on GPU 2', () => {
    expect(gpuAdvisory('0')).toMatch(/Gemma worker stays alive/);
  });

  it('warns single-GPU-2 that the Gemma model server itself is stopped', () => {
    // This is the safety-relevant case: GPU 2 is where Gemma's vLLM
    // server lives, so a lone GPU-2 training claim must say the model
    // server (not just its worker) is stopped, matching the backend
    // arbiter fix (needs_gemma_stop now returns True for '2' alone).
    const msg = gpuAdvisory('2');
    expect(msg).toMatch(/Gemma's model server/);
    expect(msg).toMatch(/stopped/);
  });

  it('describes the dual-GPU claim stopping both SAM3 and Gemma', () => {
    expect(gpuAdvisory('0,2')).toMatch(/SAM3 and Gemma/);
  });

  it('returns null for an unrecognized value rather than throwing', () => {
    expect(gpuAdvisory('7')).toBeNull();
  });
});
