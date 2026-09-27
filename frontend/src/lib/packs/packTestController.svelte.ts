/**
 * Test-on-crop for one pack (`POST /prompt_packs/test`, W5;
 * any_domain_plan.md §5.1, §7.5, §7.6 item 3; docs/design/
 * w3-pack-editor-ui-plan-2026-09-27.md §3).
 *
 * Sends exactly what the operator chose: the unsaved draft, or the saved
 * pack at the revision on screen; the call; the crop ids; and
 * `use_region_box` only when picked. Everything else takes the server's
 * default (registry classes, active profile, active VLM endpoint). The
 * response is rendered as served; nothing here parses a reply.
 */
import { untrack } from 'svelte';
import { packErrorDetail, packErrorText, testPromptPack } from '$lib/api';
import type { Crop } from '$lib/types';
import type {
  PackTestRequest,
  PackTestResponse,
  PromptPackBody,
  ValidationReport,
} from '$lib/types_packs';

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
  call = $state('');
  cropIdsText = $state('');
  source = $state<PackTestSource>('draft');
  /** `''` = not sent (the server's default applies). */
  useRegionBox = $state<'' | 'current' | 'none'>('');

  running = $state(false);
  result = $state<PackTestResponse<Crop> | null>(null);
  error = $state<string | null>(null);
  errorReport = $state<ValidationReport | null>(null);

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

  get canRun(): boolean {
    return !this.running && this.call !== '' && this.cropIds.length > 0;
  }

  request(ctx: PackTestContext): PackTestRequest {
    const req: PackTestRequest =
      this.source === 'draft'
        ? { draft: ctx.draft, call: this.call, crop_ids: this.cropIds }
        : {
            pack_name: ctx.name,
            pack_revision: ctx.revision,
            call: this.call,
            crop_ids: this.cropIds,
          };
    if (this.useRegionBox) req.use_region_box = this.useRegionBox;
    return req;
  }

  async run(ctx: PackTestContext): Promise<void> {
    if (!this.canRun) return;
    this.#abort?.abort();
    const ctl = new AbortController();
    this.#abort = ctl;
    this.running = true;
    this.error = null;
    this.errorReport = null;
    try {
      this.result = await this.#test(this.request(ctx), ctl.signal);
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.result = null;
      this.error = packErrorText(e);
      this.errorReport = packErrorDetail(e)?.report ?? null;
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
