/**
 * Test-on-crop for one pack (`POST /prompt_packs/test`).
 *
 * Sends exactly what the operator chose: the unsaved draft, or the saved
 * pack at the revision on screen; the call; the crop ids; `use_region_box`
 * only when picked; and the VLM selection only when one is set
 * (`vlmSelection`, written by the VLM picker). Everything else takes the
 * server's default (registry classes, active profile, active VLM
 * endpoint). The response is rendered as served; nothing here parses a
 * reply.
 */
import { testPromptPack } from '$lib/api_configTest';
import { ConfigTestRun, type TestSource } from '$lib/configTest/configTestRun.svelte';
import type { Crop } from '$lib/types';
import type {
  PackTestCall,
  PackTestRequest,
  PackTestResponse,
} from '$lib/types_configTest';
import type { PromptPackBody } from '$lib/types_packs';

export type PackTestSource = TestSource;

export interface PackTestContext {
  name: string;
  revision: number | null;
  draft: PromptPackBody;
}

/** Crop ids as typed: split on commas and whitespace, blanks dropped. */
export function parseCropIds(text: string): string[] {
  return text.split(/[\s,]+/).filter(Boolean);
}

export class PackTest extends ConfigTestRun<
  PackTestContext,
  PackTestRequest,
  PackTestResponse<Crop>
> {
  call = $state<PackTestCall | ''>('');
  cropIdsText = $state('');
  /** `''` = not sent (the server's default applies). */
  useRegionBox = $state<'' | 'current' | 'none'>('');

  constructor(test: typeof testPromptPack = testPromptPack) {
    super(test);
  }

  get cropIds(): string[] {
    return parseCropIds(this.cropIdsText);
  }

  get ready(): boolean {
    return this.call !== '' && this.cropIds.length > 0;
  }

  request(ctx: PackTestContext): PackTestRequest {
    const call = this.call as PackTestCall;
    const req: PackTestRequest =
      this.source === 'draft'
        ? { draft: ctx.draft, call, crop_ids: this.cropIds }
        : {
            pack_name: ctx.name,
            pack_revision: ctx.revision,
            call,
            crop_ids: this.cropIds,
          };
    if (this.useRegionBox) req.use_region_box = this.useRegionBox;
    if (this.vlmSelection) Object.assign(req, this.vlmSelection);
    return req;
  }
}

export function createPackTest(test: typeof testPromptPack = testPromptPack): PackTest {
  return new PackTest(test);
}
