/**
 * The project's active region profile (`GET /region_profiles/active`, axis
 * `detection_profile`) and the writes that move it: activate (the response
 * carries the served `impact`), rollback and deactivate (any_domain_plan.md
 * §4.4, §7.3, §7.6 item 4; docs/design/w4-profile-editor-ui-plan-2026-09-27.md
 * §3, §4).
 *
 * After every successful write the scoped `/health` is re-polled, so
 * `regionProfileStore.observe()` sees the new served profile and raises
 * the app's existing "reload the page to apply it" notice. Nothing swaps
 * the region slot in place.
 */
import {
  activateRegionProfile,
  deactivateRegionProfile,
  getActiveRegionProfile,
  rollbackRegionProfile,
} from '$lib/api';
import { ConfigActive } from '$lib/config/configActive.svelte';
import { healthStore } from '$stores/health.svelte';
import type { ProfileActivateResponse } from '$lib/types_profiles';

export interface ProfileActiveDeps {
  getActiveRegionProfile: typeof getActiveRegionProfile;
  activateRegionProfile: typeof activateRegionProfile;
  rollbackRegionProfile: typeof rollbackRegionProfile;
  deactivateRegionProfile: typeof deactivateRegionProfile;
  /** After a successful write (default: `healthStore.poll()`). */
  onchanged: () => void;
}

export class ProfileActive extends ConfigActive<ProfileActivateResponse> {
  constructor(deps: Partial<ProfileActiveDeps> = {}) {
    super({
      getActive: () => (deps.getActiveRegionProfile ?? getActiveRegionProfile)(),
      activate: (name, body) =>
        (deps.activateRegionProfile ?? activateRegionProfile)(name, body),
      rollback: (body) => (deps.rollbackRegionProfile ?? rollbackRegionProfile)(body),
      deactivate: (body) =>
        (deps.deactivateRegionProfile ?? deactivateRegionProfile)(body),
    });
    this.onchanged = deps.onchanged ?? (() => void healthStore.poll());
  }
}
