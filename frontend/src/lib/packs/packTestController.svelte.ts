/**
 * Test-on-crop for one pack (`POST /prompt_packs/test`, W5;
 * any_domain_plan.md §5.1, §7.5, §7.6 item 3; docs/design/
 * w3-pack-editor-ui-plan-2026-09-27.md §3).
 *
 * Sends exactly what the operator chose: the unsaved draft, or the saved
 * pack at the revision on screen; the call; the crop ids; `use_region_box`
 * only when picked; and the VLM selection only when one is set
 * (`vlmSelection`, written by the VLM picker). Everything else takes the
 * server's default (registry classes, active profile, active VLM
 * endpoint). The response is rendered as served; nothing here parses a
 * reply.
 */
import { untrack } from 'svelte';
import { configErrorDetail, apiErrorText } from '$lib/api';
import { testPromptPack } from '$lib/api_configTest';
import type { Crop } from '$lib/types';
import type {
  PackTestCall,
  PackTestRequest,
  PackTestResponse,
  TestVlmSelection,
} from '$lib/types_configTest';
import type { PromptPackBody } from '$lib/types_packs';
import type { ValidationReport } from '$lib/types_config';

export type PackTestSource = 'draft' | 'saved';

export interface PackTestContext {
  name: string;
  revision: number | null;
  draft: PromptPackBody;
}

/** Crop ids as typed: split on commas and whitespace, blanks dropped. */
export function parseCropIds(text: string): string[] {
  return text.split(/[\s,]+/).filter(Boolean);
}

export class PackTest {
  call = $state<PackTestCall | ''>('');
  cropIdsText = $state('');
  source = $state<PackTestSource>('draft');
  /** `''` = not sent (the server's default applies). */
  useRegionBox = $state<'' | 'current' | 'none'>('');
  /** The VLM to answer, when the operator picked one; null = the server's
   *  default (the active endpoint). */
  vlmSelection = $state<TestVlmSelection | null>(null);

  running = $state(false);
  result = $state<PackTestResponse<Crop> | null>(null);
  error = $state<string | null>(null);
  errorReport = $state<ValidationReport | null>(null);
  /** Ids the server named in a `crop_not_found` refusal. */
  missingCropIds = $state<string[]>([]);

  #test: typeof testPromptPack;
  #abort: AbortController | null = null;

  constructor(test: typeof testPromptPack = testPromptPack) {
    this.#test = test;
  }

  #forcedFrom: PackTestSource | null = null;

  /** No draft to test (a read-only pack, or a revision being viewed):
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

  get cropIds(): string[] {
    return parseCropIds(this.cropIdsText);
  }

  /** Enough input to run (a rerun while one is in flight aborts it). */
  get ready(): boolean {
    return this.call !== '' && this.cropIds.length > 0;
  }

  /** What the Run button enables on. */
  get canRun(): boolean {
    return !this.running && this.ready;
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

  async run(ctx: PackTestContext): Promise<void> {
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

export function createPackTest(test: typeof testPromptPack = testPromptPack): PackTest {
  return new PackTest(test);
}
