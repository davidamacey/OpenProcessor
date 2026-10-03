/**
 * One region profile's editor state (any_domain_plan.md §4.3, §4.4, §7.3,
 * §7.4, §7.6 items 2 and 4; docs/design/w4-profile-editor-ui-plan-2026-09-27.md
 * §4): the profile binding of the shared `ConfigEditor` (draft, served
 * live validation, Save with `expected_revision`, revisions and restore,
 * the `config.changed axis=detection_profile` wake-up), plus
 *
 * - the served vocabulary every model picker renders from, re-read with
 *   `include_other_projects` on request (projects_plan.md §5.5);
 * - "Check for activation": the draft posted with `for_activation=true`,
 *   its served report kept apart from the live one;
 * - "Check segmenter prompt": the draft's prompt posted to the text-only
 *   route, its served report kept apart and cleared by the next edit.
 */
import {
  apiErrorText,
  getConfigVocabulary,
  getRegionProfile,
  getRegionProfileRevision,
  getRegionProfileRevisions,
  getRegionProfileSchema,
  updateRegionProfile,
  validateRegionProfile,
  validateSegmenterPrompt,
} from '$lib/api';
import { ConfigEditor } from '$lib/config/configEditor.svelte';
import { subscribeCurationEvents, type CurationEvent } from '$lib/sse';
import type { ValidationReport } from '$lib/types_config';
import type {
  ConfigVocabulary,
  RegionProfileBody,
  RegionProfileDoc,
  RegionProfileSchema,
} from '$lib/types_profiles';
import { ProfileActive, type ProfileActiveDeps } from './profileActive.svelte';
import { isProfileConfigEvent } from './profileListController.svelte';

export interface ProfileEditorDeps extends ProfileActiveDeps {
  getRegionProfile: typeof getRegionProfile;
  getRegionProfileSchema: typeof getRegionProfileSchema;
  getRegionProfileRevisions: typeof getRegionProfileRevisions;
  getRegionProfileRevision: typeof getRegionProfileRevision;
  updateRegionProfile: typeof updateRegionProfile;
  validateRegionProfile: typeof validateRegionProfile;
  validateSegmenterPrompt: typeof validateSegmenterPrompt;
  getConfigVocabulary: typeof getConfigVocabulary;
  subscribe: (onEvent: (e: CurationEvent) => void) => { close(): void };
}

export class ProfileEditor extends ConfigEditor<
  RegionProfileBody,
  RegionProfileDoc,
  RegionProfileSchema,
  ProfileActive
> {
  vocabulary = $state<ConfigVocabulary | null>(null);
  vocabularyError = $state<string | null>(null);
  /** Whether the vocabulary lists other projects' shared detectors. */
  includeOtherProjects = $state(false);

  /** The served report of the last "Check for activation". */
  activationReport = $state<ValidationReport | null>(null);
  activationChecking = $state(false);
  activationCheckError = $state<string | null>(null);

  #deps: Partial<ProfileEditorDeps>;

  constructor(name: string, deps: Partial<ProfileEditorDeps> = {}) {
    super(
      name,
      {
        getSchema: () => (deps.getRegionProfileSchema ?? getRegionProfileSchema)(),
        getDoc: (n) => (deps.getRegionProfile ?? getRegionProfile)(n),
        getRevisions: (n) =>
          (deps.getRegionProfileRevisions ?? getRegionProfileRevisions)(n),
        getRevision: (n, r) =>
          (deps.getRegionProfileRevision ?? getRegionProfileRevision)(n, r),
        update: (n, body) => (deps.updateRegionProfile ?? updateRegionProfile)(n, body),
        validate: (body, signal) =>
          (deps.validateRegionProfile ?? validateRegionProfile)(body, false, signal),
        subscribe:
          deps.subscribe ??
          ((onEvent) => subscribeCurationEvents({ topic: 'config', onEvent })),
        isEvent: isProfileConfigEvent,
      },
      new ProfileActive(deps),
    );
    this.#deps = deps;
  }

  protected override async loadExtras(): Promise<void> {
    await this.loadVocabulary();
  }

  /** Reads the served vocabulary. A failure is shown, never fatal: the
   *  pickers then fall back to the stored value alone. */
  async loadVocabulary(): Promise<void> {
    try {
      this.vocabulary = await (this.#deps.getConfigVocabulary ?? getConfigVocabulary)(
        this.includeOtherProjects,
      );
      this.vocabularyError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.vocabularyError = apiErrorText(e);
    }
  }

  async setIncludeOtherProjects(on: boolean): Promise<void> {
    if (this.includeOtherProjects === on) return;
    this.includeOtherProjects = on;
    await this.loadVocabulary();
  }

  /** Posts the draft with `for_activation=true` (§7.6 item 2). Never
   *  blocks anything; the served report is shown apart. */
  async checkForActivation(): Promise<void> {
    this.activationChecking = true;
    try {
      this.activationReport = await (
        this.#deps.validateRegionProfile ?? validateRegionProfile
      )({ name: null, body: this.draftBody }, true);
      this.activationCheckError = null;
    } catch (e) {
      this.activationCheckError = apiErrorText(e);
    } finally {
      this.activationChecking = false;
    }
  }

  promptReport = $state<ValidationReport | null>(null);
  promptChecking = $state(false);
  promptCheckError = $state<string | null>(null);

  /** `POST /region_profiles/validate_segmenter_prompt` for the draft's
   *  prompt. `sole_leg` mirrors the server's own call: no detector means
   *  the segmenter is the only candidate source. */
  async checkSegmenterPrompt(): Promise<void> {
    this.promptChecking = true;
    try {
      this.promptReport = await (
        this.#deps.validateSegmenterPrompt ?? validateSegmenterPrompt
      )({
        text_prompt: String(this.draftBody?.segmenter_text_prompt ?? ''),
        sole_leg: !this.draftBody?.detector_model,
      });
      this.promptCheckError = null;
    } catch (e) {
      this.promptReport = null;
      this.promptCheckError = apiErrorText(e);
    } finally {
      this.promptChecking = false;
    }
  }

  override setField(field: string, value: RegionProfileBody[string]): void {
    super.setField(field, value);
    // The check was for a different draft.
    this.activationReport = null;
    this.promptReport = null;
  }
}

export function createProfileEditor(
  name: string,
  deps: Partial<ProfileEditorDeps> = {},
): ProfileEditor {
  return new ProfileEditor(name, deps);
}
