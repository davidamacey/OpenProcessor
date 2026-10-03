/**
 * Test-on-one-image for an unsaved open-vocabulary target
 * (`POST /open_vocab/test`; nothing is written).
 *
 * The route takes `image_id` or `image_base64`, never a crop id. The crop
 * source reads that crop and sends its served `image_id`; the upload source
 * sends the file's base64 as is (no client resize). `gating` is sent only
 * when the VLM pre-check is ticked; `image_max_side` / `dedup_iou` come from
 * the draft's own values. A new run aborts the one in flight. A 502
 * `segmenter_error` is an error, never "no hits"; every refusal reads as
 * served. Nothing here draws or judges a hit.
 */
import { ApiError, configErrorDetail, apiErrorText, getCrop } from '$lib/api';
import { testOpenVocab } from '$lib/api_openVocab';
import type {
  OpenVocabTargetBody,
  OpenVocabTestRequest,
  OpenVocabTestResponse,
} from '$lib/types_openVocab';

export type OpenVocabTestSource = 'crop' | 'upload';

export interface OpenVocabTestContext {
  target: OpenVocabTargetBody;
  image_max_side?: number;
  dedup_iou?: number;
}

export interface OpenVocabTestDeps {
  getCrop: typeof getCrop;
  testOpenVocab: typeof testOpenVocab;
}

export class OpenVocabTest {
  source = $state<OpenVocabTestSource>('crop');
  cropId = $state('');
  upload = $state<{ name: string; base64: string } | null>(null);
  /** "Run the VLM pre-check": sends `gating.tier2_vlm_precheck` only when ticked. */
  precheck = $state(false);

  running = $state(false);
  result = $state<OpenVocabTestResponse | null>(null);
  error = $state<string | null>(null);
  /** The error is the 502 `segmenter_error` (shown as a segmenter failure). */
  segmenterError = $state(false);
  /** The image the last successful run used, for the overlay: the crop id
   *  whose image it was, or null for an upload. */
  ranOnCropId = $state<string | null>(null);

  #deps: OpenVocabTestDeps;
  #abort: AbortController | null = null;

  constructor(deps: Partial<OpenVocabTestDeps> = {}) {
    this.#deps = { getCrop, testOpenVocab, ...deps };
  }

  get ready(): boolean {
    return this.source === 'crop' ? this.cropId.trim() !== '' : this.upload != null;
  }

  get canRun(): boolean {
    return !this.running && this.ready;
  }

  async run(ctx: OpenVocabTestContext): Promise<void> {
    if (!this.ready) return;
    this.#abort?.abort();
    const ctl = new AbortController();
    this.#abort = ctl;
    this.running = true;
    this.error = null;
    this.segmenterError = false;
    try {
      const req: OpenVocabTestRequest = { target: ctx.target };
      let cropId: string | null = null;
      if (this.source === 'crop') {
        cropId = this.cropId.trim();
        const imageId = await this.#imageIdOf(cropId, ctl.signal);
        if (imageId == null) return;
        req.image_id = imageId;
      } else {
        req.image_base64 = this.upload!.base64;
      }
      if (ctx.image_max_side != null) req.image_max_side = ctx.image_max_side;
      if (ctx.dedup_iou != null) req.dedup_iou = ctx.dedup_iou;
      if (this.precheck) req.gating = { tier2_vlm_precheck: true };
      const res = await this.#deps.testOpenVocab(req, ctl.signal);
      if (ctl.signal.aborted) return;
      this.result = res;
      this.ranOnCropId = cropId;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.result = null;
      this.error = apiErrorText(e);
      this.segmenterError = configErrorDetail(e)?.error === 'segmenter_error';
    } finally {
      if (this.#abort === ctl) {
        this.running = false;
        this.#abort = null;
      }
    }
  }

  /** The crop's served `image_id`, or null after setting the refusal. */
  async #imageIdOf(cropId: string, signal: AbortSignal): Promise<string | null> {
    try {
      const crop = await this.#deps.getCrop(cropId, signal);
      if (crop.image_id == null) {
        this.result = null;
        this.error = `Crop ${cropId} has no source image to test on.`;
        return null;
      }
      return crop.image_id;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return null;
      this.result = null;
      this.error =
        e instanceof ApiError && e.status === 404
          ? `Crop ${cropId} was not found.`
          : apiErrorText(e);
      return null;
    }
  }

  stop(): void {
    this.#abort?.abort();
  }
}

export function createOpenVocabTest(
  deps: Partial<OpenVocabTestDeps> = {},
): OpenVocabTest {
  return new OpenVocabTest(deps);
}
