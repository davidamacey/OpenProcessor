/**
 * Follow-up gap 2 (docs/design/audit-remediation-plan-2026-09.md
 * Appendix D item 3, 2026-09-11): before this module existed, `/models`
 * had no unload action at all — Phase 10's cleanup of a throwaway
 * promoted model required direct Triton API calls. These tests cover the
 * button-visibility/confirmation logic that gates the new action,
 * particularly that the active model can never be unloaded with a single
 * plain confirmation.
 */

import { describe, expect, it } from 'vitest';
import {
  unloadButtonState,
  unloadConfirmMessage,
  unloadForceConfirmMessage,
} from './modelUnload';

describe('unloadButtonState', () => {
  it('hides the button entirely for LPR models', () => {
    expect(
      unloadButtonState({
        kind: 'triton',
        is_region_protected: true,
        requires_force_to_unload: false,
      }),
    ).toBe('hidden');
  });

  it('hides the button entirely for LPR models even if also flagged core/active (belt and suspenders)', () => {
    expect(
      unloadButtonState({
        kind: 'triton',
        is_region_protected: true,
        requires_force_to_unload: true,
      }),
    ).toBe('hidden');
  });

  it('hides the button for non-Triton models (Gemma)', () => {
    expect(
      unloadButtonState({
        kind: 'external',
        is_region_protected: false,
        requires_force_to_unload: false,
      }),
    ).toBe('hidden');
  });

  it('requires explicit force confirmation for the active/core model — never a bare single confirm', () => {
    // This is the safety-relevant case: the active vehicle model must
    // never be unloadable via the same one-click path as a disposable
    // throwaway promote.
    expect(
      unloadButtonState({
        kind: 'triton',
        is_region_protected: false,
        requires_force_to_unload: true,
      }),
    ).toBe('force-required');
  });

  it('is a normal single-confirm action for an ordinary (non-LPR, non-core) Triton model', () => {
    expect(
      unloadButtonState({
        kind: 'triton',
        is_region_protected: false,
        requires_force_to_unload: false,
      }),
    ).toBe('normal');
  });

  it('defaults undefined flags to falsy (server omits them for kind=external)', () => {
    expect(
      unloadButtonState({
        kind: 'triton',
        is_region_protected: undefined,
        requires_force_to_unload: undefined,
      }),
    ).toBe('normal');
  });
});

describe('unloadConfirmMessage', () => {
  it('warns loudly about live traffic + permanence for a force-required model', () => {
    const msg = unloadConfirmMessage({
      name: 'yolov11_small_trt_end2end',
      requires_force_to_unload: true,
    });
    expect(msg).toMatch(/live traffic/);
    expect(msg).toMatch(/cannot be undone/);
  });

  it('is a plain permanence warning for a normal model', () => {
    const msg = unloadConfirmMessage({
      name: 'op_vehicle_smoke_v1',
      requires_force_to_unload: false,
    });
    expect(msg).toMatch(/op_vehicle_smoke_v1/);
    expect(msg).toMatch(/cannot be undone/);
    expect(msg).not.toMatch(/live traffic/);
  });
});

describe('unloadForceConfirmMessage', () => {
  it('names the model in the second confirmation', () => {
    expect(unloadForceConfirmMessage({ name: 'legacy_vehicle_v6_trt' })).toMatch(
      /legacy_vehicle_v6_trt/,
    );
  });
});
