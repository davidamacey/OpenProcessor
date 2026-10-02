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
  showsProtectedChip,
  unloadButtonState,
  unloadConfirmMessage,
  unloadForceConfirmMessage,
} from './modelUnload';

// GET {API_PREFIX}/models/status serves `unloadable` on every entry
// (false for the external segmenter/VLM), and `is_region_protected` covers
// every model the active config hard-blocks (403, no force flow).
describe('unloadButtonState', () => {
  it('hides the button whenever the server says unloadable: false, whatever the flags say', () => {
    expect(
      unloadButtonState({
        is_region_protected: false,
        requires_force_to_unload: false,
        unloadable: false,
      }),
    ).toBe('hidden');
    // An external entry (the segmenter, the VLM).
    expect(
      unloadButtonState({
        is_region_protected: undefined,
        requires_force_to_unload: undefined,
        unloadable: false,
      }),
    ).toBe('hidden');
  });

  it('hides the button for region-protected models even when unloadable: true', () => {
    expect(
      unloadButtonState({
        is_region_protected: true,
        requires_force_to_unload: false,
        unloadable: true,
      }),
    ).toBe('hidden');
    expect(
      unloadButtonState({
        is_region_protected: true,
        requires_force_to_unload: true,
        unloadable: true,
      }),
    ).toBe('hidden');
  });

  it('requires explicit force confirmation for the active/core model — never a bare single confirm', () => {
    // The active production model must never be unloadable via the same
    // one-click path as a disposable throwaway promote.
    expect(
      unloadButtonState({
        is_region_protected: false,
        requires_force_to_unload: true,
        unloadable: true,
      }),
    ).toBe('force-required');
  });

  it('is a normal single-confirm action for an ordinary unloadable model', () => {
    expect(
      unloadButtonState({
        is_region_protected: false,
        requires_force_to_unload: false,
        unloadable: true,
      }),
    ).toBe('normal');
    expect(
      unloadButtonState({
        is_region_protected: undefined,
        requires_force_to_unload: undefined,
        unloadable: true,
      }),
    ).toBe('normal');
  });
});

describe('showsProtectedChip', () => {
  it('true for a region-protected model that IS otherwise unloadable', () => {
    expect(showsProtectedChip({ is_region_protected: true, unloadable: true })).toBe(
      true,
    );
  });

  it('false for an external service (not region-protected, just not unloadable at all)', () => {
    expect(
      showsProtectedChip({ is_region_protected: undefined, unloadable: false }),
    ).toBe(false);
  });

  it('false for an ordinary unprotected model', () => {
    expect(showsProtectedChip({ is_region_protected: false, unloadable: true })).toBe(
      false,
    );
  });

  it('false for protected AND explicitly not unloadable — matches the button being hidden either way', () => {
    expect(showsProtectedChip({ is_region_protected: true, unloadable: false })).toBe(
      false,
    );
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
      name: 'vehicle_smoke_v1',
      requires_force_to_unload: false,
    });
    expect(msg).toMatch(/vehicle_smoke_v1/);
    expect(msg).toMatch(/cannot be undone/);
    expect(msg).not.toMatch(/live traffic/);
  });
});

describe('unloadForceConfirmMessage', () => {
  it('names the model in the second confirmation', () => {
    expect(unloadForceConfirmMessage({ name: 'vehicle_classifier_trt' })).toMatch(
      /vehicle_classifier_trt/,
    );
  });
});
