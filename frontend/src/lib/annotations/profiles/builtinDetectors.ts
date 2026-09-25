/**
 * This deployment's outcome/tag muting config for `ProvenanceChip.svelte`.
 *
 * Labels and chip colors used to be hardcoded name→label and
 * name/prefix→palette tables here (`DetectorChip.svelte`'s old
 * `labelFor`/`paletteFor`). Both are now served by the backend
 * (`GET {API_PREFIX}/regions/vocabulary`, W0 naming-sweep finding m9) — labels from
 * the vocabulary entry itself, colors from its `role` via
 * `detectorRegistry.ts`'s `paletteForRole`. This file only keeps what's
 * genuinely still a deployment concern: which chain-entry *tags*
 * (outcomes, not detector ids) render muted.
 */

import type { MutedTagConfig } from '../detectorRegistry';

export const builtinDetectorRegistry: MutedTagConfig = {
  // `accepted_unverified` (2026-09-24 logic-moves W8): a chain step the
  // backend accepted without a human/VLM verification pass — muted,
  // same as a miss/reject, so it reads as "lower confidence" rather than
  // a confirmed step.
  mutedTagPattern: /miss|reject|skipped|degenerate|unparseable|accepted_unverified/,
};
