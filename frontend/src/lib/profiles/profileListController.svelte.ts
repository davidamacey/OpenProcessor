/**
 * The region-profile list page's state (any_domain_plan.md §7.3, §7.6
 * items 2 and 4; docs/design/w4-profile-editor-ui-plan-2026-09-27.md §3):
 * the profile binding of the shared `ConfigList` (the served list and
 * templates, the active profile with Rollback and Turn off, Clone and
 * Delete), plus the served activation impact on demand. A
 * `config.changed axis=detection_profile` event is a wake-up to re-read.
 */
import {
  cloneRegionProfile,
  apiErrorText,
  deleteRegionProfile,
  getConfigVocabulary,
  getRegionProfileImpact,
  listRegionProfiles,
} from '$lib/api';
import { ConfigList } from '$lib/config/configList.svelte';
import { isConfigAxisEvent } from '$lib/config/validationIssues';
import { subscribeCurationEvents, type CurationEvent } from '$lib/sse';
import type {
  ActivationImpact,
  ConfigVocabulary,
  RegionProfileDoc,
  RegionProfileList,
} from '$lib/types_profiles';
import { ProfileActive, type ProfileActiveDeps } from './profileActive.svelte';

/** §7.1: the region-profile axis id on every surface. */
export const PROFILE_AXIS = 'detection_profile';

export function isProfileConfigEvent(e: CurationEvent): boolean {
  return isConfigAxisEvent(e, PROFILE_AXIS);
}

export interface ProfileListDeps extends ProfileActiveDeps {
  listRegionProfiles: typeof listRegionProfiles;
  cloneRegionProfile: typeof cloneRegionProfile;
  deleteRegionProfile: typeof deleteRegionProfile;
  getRegionProfileImpact: typeof getRegionProfileImpact;
  getConfigVocabulary: typeof getConfigVocabulary;
  subscribe: (onEvent: (e: CurationEvent) => void) => { close(): void };
}

export class ProfileList extends ConfigList<
  RegionProfileList,
  RegionProfileDoc,
  ProfileActive
> {
  impact = $state<ActivationImpact | null>(null);
  impactError = $state<string | null>(null);
  impactLoading = $state(false);
  vocabulary = $state<ConfigVocabulary | null>(null);
  vocabularyError = $state<string | null>(null);
  vocabularyLoading = $state(false);

  #getImpact: () => Promise<ActivationImpact>;
  #getVocabulary: () => Promise<ConfigVocabulary>;

  constructor(deps: Partial<ProfileListDeps> = {}) {
    super(
      {
        list: () => (deps.listRegionProfiles ?? listRegionProfiles)(),
        clone: (name, body) =>
          (deps.cloneRegionProfile ?? cloneRegionProfile)(name, body),
        remove: (name, rev) =>
          (deps.deleteRegionProfile ?? deleteRegionProfile)(name, rev),
        subscribe:
          deps.subscribe ??
          ((onEvent) => subscribeCurationEvents({ topic: 'config', onEvent })),
        isEvent: isProfileConfigEvent,
      },
      new ProfileActive(deps),
    );
    this.#getImpact = () => (deps.getRegionProfileImpact ?? getRegionProfileImpact)();
    this.#getVocabulary = () => (deps.getConfigVocabulary ?? getConfigVocabulary)(false);
  }

  /** Turns region detection off; re-reads the list on success. A shown
   *  impact was for the previous activation, so it is dropped. */
  async deactivate(): Promise<boolean> {
    const ok = await this.active.deactivate();
    if (ok) {
      this.impact = null;
      await this.load();
    }
    return ok;
  }

  override async rollback(): Promise<boolean> {
    const ok = await super.rollback();
    if (ok) this.impact = null;
    return ok;
  }

  /** Reads the served vocabulary for the read-only "Models and sources"
   *  panel (loaded when the panel is first opened). */
  async loadVocabulary(): Promise<void> {
    if (this.vocabulary || this.vocabularyLoading) return;
    this.vocabularyLoading = true;
    try {
      this.vocabulary = await this.#getVocabulary();
      this.vocabularyError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.vocabularyError = apiErrorText(e);
    } finally {
      this.vocabularyLoading = false;
    }
  }

  /** Reads the served impact of the current activation (§4.6). */
  async loadImpact(): Promise<void> {
    this.impactLoading = true;
    try {
      this.impact = await this.#getImpact();
      this.impactError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.impactError = apiErrorText(e);
    } finally {
      this.impactLoading = false;
    }
  }
}

export function createProfileList(deps: Partial<ProfileListDeps> = {}): ProfileList {
  return new ProfileList(deps);
}
