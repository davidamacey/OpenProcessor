/**
 * Test-on-crop for one region profile (`POST /region_profiles/test`, W5;
 * any_domain_plan.md §5.2, §7.6, §7.7).
 *
 * Sends exactly what the operator chose: one crop id; the unsaved draft,
 * or the saved profile at the revision on screen; the segmenter prompt
 * override only when non-empty; `verify: true` only when ticked; and the
 * VLM selection only when one is set. Everything else takes the server's
 * default. The response is rendered as served; nothing here projects a
 * box or judges a candidate.
 */
import { untrack } from 'svelte';
import { configErrorDetail, apiErrorText } from '$lib/api';
import { testRegionProfile } from '$lib/api_configTest';
import type { ValidationReport } from '$lib/types_config';
import type { RegionProfileBody } from '$lib/types_profiles';
import type {
  RegionTestRequest,
  RegionTestResponse,
  TestVlmSelection,
} from '$lib/types_configTest';

export type ProfileTestSource = 'draft' | 'saved';

export interface ProfileTestContext {
  name: string;
  revision: number | null;
  draft: RegionProfileBody;
}

export class ProfileTest {
  cropId = $state('');
  source = $state<ProfileTestSource>('draft');
  /** Free text; sent only when non-empty. */
  segmenterPrompt = $state('');
  verify = $state(false);
  /** The VLM to answer a verify, when the operator picked one; null =
   *  the server's default (the active endpoint). */
  vlmSelection = $state<TestVlmSelection | null>(null);

  running = $state(false);
  result = $state<RegionTestResponse | null>(null);
  error = $state<string | null>(null);
  errorReport = $state<ValidationReport | null>(null);
  /** Ids the server named in a `crop_not_found` refusal. */
  missingCropIds = $state<string[]>([]);

  #test: typeof testRegionProfile;
  #abort: AbortController | null = null;
  #forcedFrom: ProfileTestSource | null = null;

  constructor(test: typeof testRegionProfile = testRegionProfile) {
    this.#test = test;
  }

  /** No draft to test (a read-only profile, or a revision being viewed):
   *  forces the saved source; when a draft is back, restores what the
   *  operator had before, never overriding a choice they made. */
  setSavedOnly(savedOnly: boolean): void {
    if (savedOnly && this.#forcedFrom == null) {
      this.#forcedFrom = untrack(() => this.source);
      this.source = 'saved';
    } else if (!savedOnly && this.#forcedFrom != null) {
      this.source = this.#forcedFrom;
      this.#forcedFrom = null;
    }
  }

  /** Enough input to run (a rerun while one is in flight aborts it). */
  get ready(): boolean {
    return this.cropId.trim() !== '';
  }

  /** What the Run button enables on. */
  get canRun(): boolean {
    return !this.running && this.ready;
  }

  request(ctx: ProfileTestContext): RegionTestRequest {
    const crop_id = this.cropId.trim();
    const req: RegionTestRequest =
      this.source === 'draft'
        ? { crop_id, draft: ctx.draft }
        : { crop_id, profile_name: ctx.name, profile_revision: ctx.revision };
    if (this.segmenterPrompt.trim() !== '') {
      req.segmenter_text_prompt = this.segmenterPrompt;
    }
    if (this.verify) req.verify = true;
    // The VLM only answers the verify pass.
    if (this.verify && this.vlmSelection) Object.assign(req, this.vlmSelection);
    return req;
  }

  async run(ctx: ProfileTestContext): Promise<void> {
    if (!this.ready) return;
    this.#abort?.abort();
    const ctl = new AbortController();
    this.#abort = ctl;
    this.running = true;
    this.error = null;
    this.errorReport = null;
    this.missingCropIds = [];
    try {
      this.result = await this.#test(this.request(ctx), ctl.signal);
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.result = null;
      this.error = apiErrorText(e);
      const d = configErrorDetail(e);
      this.errorReport = d?.report ?? null;
      this.missingCropIds = d?.error === 'crop_not_found' ? (d.crop_ids ?? []) : [];
    } finally {
      if (this.#abort === ctl) {
        this.running = false;
        this.#abort = null;
      }
    }
  }

  stop(): void {
    this.#abort?.abort();
  }
}

export function createProfileTest(
  test: typeof testRegionProfile = testRegionProfile,
): ProfileTest {
  return new ProfileTest(test);
}
