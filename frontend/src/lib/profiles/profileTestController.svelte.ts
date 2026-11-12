/**
 * Test-on-crop for one region profile (`POST /region_profiles/test`).
 *
 * Sends exactly what the operator chose: one crop id; the unsaved draft,
 * or the saved profile at the revision on screen; the segmenter prompt
 * override only when non-empty; `verify: true` only when ticked; and the
 * VLM selection only when one is set. Everything else takes the server's
 * default. The response is rendered as served; nothing here projects a
 * box or judges a candidate.
 */
import { testRegionProfile } from '$lib/api_configTest';
import { ConfigTestRun, type TestSource } from '$lib/configTest/configTestRun.svelte';
import type { RegionProfileBody } from '$lib/types_profiles';
import type { RegionTestRequest, RegionTestResponse } from '$lib/types_configTest';

export type ProfileTestSource = TestSource;

export interface ProfileTestContext {
  name: string;
  revision: number | null;
  draft: RegionProfileBody;
}

export class ProfileTest extends ConfigTestRun<
  ProfileTestContext,
  RegionTestRequest,
  RegionTestResponse
> {
  cropId = $state('');
  /** Free text; sent only when non-empty. */
  segmenterPrompt = $state('');
  verify = $state(false);

  constructor(test: typeof testRegionProfile = testRegionProfile) {
    super(test);
  }

  get ready(): boolean {
    return this.cropId.trim() !== '';
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
}

export function createProfileTest(
  test: typeof testRegionProfile = testRegionProfile,
): ProfileTest {
  return new ProfileTest(test);
}
