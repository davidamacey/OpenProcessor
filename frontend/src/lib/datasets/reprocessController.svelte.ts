/**
 * Reprocess dialog state (any_domain_plan.md §7.12 item 6, W10.13;
 * docs/design/w10-import-reprocess-ui-plan-2026-09-27.md §5).
 *
 * - One crop: `POST /crops/{id}/reprocess` with `dry_run: false` (the
 *   route's own default; the dialog is the confirm step). The served
 *   post-write items are handed back for the host to adopt.
 * - One image: `POST /images/{id}/reprocess`, same single-target flow;
 *   every served item of that image is handed back.
 * - Several crops: `POST /reprocess` with `dry_run: true` first (served
 *   per-scope `selected` / `locked_skipped` / `breakdown`),
 *   then the same request with `dry_run: false`. A served `job` is
 *   followed at `GET /reprocess/jobs/{job_id}`.
 * - A served request (W4's activation `impact.suggested_reprocess`,
 *   docs/design/w4-profile-editor-ui-plan-2026-09-27.md §5): the same
 *   dry-run-then-apply, sending the request exactly as served; its scopes
 *   and region mode can't be changed.
 *
 * Human and imported labels are locked server-side; the client never
 * words an outcome itself, only the served counts are shown.
 */
import {
  cancelReprocessJob,
  apiErrorText,
  getReprocessJob,
  reprocessBatch,
  reprocessCrop,
  reprocessImage,
} from '$lib/api';
import type { Crop } from '$lib/types';
import type {
  EmbedOptions,
  ReprocessJob,
  ReprocessRegionMode,
  ReprocessRequest,
  ReprocessResponse,
  ReprocessScope,
} from '$lib/types_import';

export type EmbedPart = NonNullable<EmbedOptions['parts']>[number];

export type ReprocessTarget =
  | { kind: 'crop'; cropId: string }
  | { kind: 'image'; imageId: string }
  | { kind: 'crops'; cropIds: string[] }
  | { kind: 'request'; request: ReprocessRequest };

export interface ReprocessDeps {
  reprocessBatch: typeof reprocessBatch;
  reprocessCrop: typeof reprocessCrop;
  reprocessImage: typeof reprocessImage;
  getReprocessJob: typeof getReprocessJob;
  cancelReprocessJob: typeof cancelReprocessJob;
}

// Resolved when a dialog opens, not at import time.
const defaultDeps = (): ReprocessDeps => ({
  reprocessBatch,
  reprocessCrop,
  reprocessImage,
  getReprocessJob,
  cancelReprocessJob,
});

export class ReprocessFlow {
  readonly target: ReprocessTarget;
  scopes = $state<ReprocessScope[]>([]);
  /** '' = the server's default region mode. */
  regionMode = $state<ReprocessRegionMode | ''>('');
  /** Embed options: null = untouched, so the server's default applies and
   *  the key is not sent. */
  embedOnlyMissing = $state<boolean | null>(null);
  embedParts = $state<EmbedPart[] | null>(null);
  /** The batch dry run for the current choices; null when stale. */
  dryRun = $state<ReprocessResponse | null>(null);
  result = $state<ReprocessResponse | null>(null);
  job = $state<ReprocessJob | null>(null);
  error = $state<string | null>(null);
  busy = $state(false);

  #deps: ReprocessDeps;
  #timer: ReturnType<typeof setTimeout> | null = null;

  constructor(target: ReprocessTarget, deps: Partial<ReprocessDeps> = {}) {
    this.target = target;
    this.#deps = { ...defaultDeps(), ...deps };
    if (target.kind === 'request') this.scopes = [...target.request.scopes];
  }

  get isBatch(): boolean {
    return this.target.kind === 'crops' || this.target.kind === 'request';
  }

  /** The number of crops targeted; 0 for a served request, whose size is
   *  only known from its dry run. */
  get count(): number {
    if (this.target.kind === 'crop' || this.target.kind === 'image') return 1;
    if (this.target.kind === 'crops') return this.target.cropIds.length;
    return 0;
  }

  toggleScope(id: ReprocessScope, on: boolean): void {
    if (this.target.kind === 'request') return;
    this.scopes = on
      ? [...this.scopes.filter((s) => s !== id), id]
      : this.scopes.filter((s) => s !== id);
    this.#invalidate();
  }

  setRegionMode(mode: ReprocessRegionMode | ''): void {
    if (this.target.kind === 'request') return;
    this.regionMode = mode;
    this.#invalidate();
  }

  /** Embed options can only be set on the batch routes (the one-crop and
   *  one-image bodies have no `embed`), and only for the crops the operator
   *  chose; a served request keeps its own. */
  get embedOptionsEditable(): boolean {
    return this.target.kind === 'crops' && this.scopes.includes('embed');
  }

  setEmbedOnlyMissing(on: boolean): void {
    if (this.target.kind === 'request') return;
    this.embedOnlyMissing = on;
    this.#invalidate();
  }

  toggleEmbedPart(part: EmbedPart, on: boolean): void {
    if (this.target.kind === 'request') return;
    const cur = this.embedParts ?? [];
    const next = on
      ? [...cur.filter((p) => p !== part), part]
      : cur.filter((p) => p !== part);
    this.embedParts = next.length > 0 ? next : null;
    this.#invalidate();
  }

  #embedOptions(): { embed?: EmbedOptions } {
    if (!this.scopes.includes('embed')) return {};
    const embed: EmbedOptions = {};
    if (this.embedOnlyMissing !== null) embed.only_missing = this.embedOnlyMissing;
    if (this.embedParts !== null) embed.parts = this.embedParts;
    return Object.keys(embed).length > 0 ? { embed } : {};
  }

  #invalidate(): void {
    this.dryRun = null;
    this.result = null;
    this.error = null;
  }

  #regionMode(): { region_mode?: ReprocessRegionMode } {
    return this.scopes.includes('region') && this.regionMode
      ? { region_mode: this.regionMode }
      : {};
  }

  /** The batch request: a served one as served, else the chosen crops. */
  #batchRequest(dryRun: boolean): ReprocessRequest | null {
    if (this.target.kind === 'request')
      return { ...this.target.request, dry_run: dryRun };
    if (this.target.kind !== 'crops') return null;
    return {
      targets: { crop_ids: this.target.cropIds },
      scopes: this.scopes,
      ...this.#regionMode(),
      ...this.#embedOptions(),
      dry_run: dryRun,
    };
  }

  /** Batch only: what would run, as served. Writes nothing. */
  async preview(): Promise<void> {
    const body = this.#batchRequest(true);
    if (!body || this.scopes.length === 0 || this.busy) return;
    this.busy = true;
    this.error = null;
    try {
      this.dryRun = await this.#deps.reprocessBatch(body);
    } catch (e) {
      this.error = apiErrorText(e);
    } finally {
      this.busy = false;
    }
  }

  get canApply(): boolean {
    if (this.scopes.length === 0 || this.busy || this.result) return false;
    return this.isBatch ? this.dryRun != null : true;
  }

  /** Apply. Returns the served post-write crops (single target) to adopt. */
  async apply(): Promise<Crop[]> {
    if (!this.canApply) return [];
    this.busy = true;
    this.error = null;
    try {
      if (this.target.kind === 'crop' || this.target.kind === 'image') {
        const body = {
          scopes: this.scopes,
          ...this.#regionMode(),
          dry_run: false,
        };
        const res =
          this.target.kind === 'crop'
            ? await this.#deps.reprocessCrop(this.target.cropId, body)
            : await this.#deps.reprocessImage(this.target.imageId, body);
        this.result = res;
        return res.items ?? [];
      }
      const res = await this.#deps.reprocessBatch(this.#batchRequest(false)!);
      this.result = res;
      if (res.job) this.#follow(res.job);
      return [];
    } catch (e) {
      this.error = apiErrorText(e);
      return [];
    } finally {
      this.busy = false;
    }
  }

  #follow(job: ReprocessJob): void {
    this.job = job;
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
    if (job.poll_after_s == null) return;
    this.#timer = setTimeout(async () => {
      try {
        this.#follow(await this.#deps.getReprocessJob(job.job_id));
      } catch (e) {
        this.error = apiErrorText(e);
      }
    }, job.poll_after_s * 1000);
  }

  async cancelJob(): Promise<void> {
    if (!this.job) return;
    try {
      this.#follow(await this.#deps.cancelReprocessJob(this.job.job_id));
    } catch (e) {
      this.error = apiErrorText(e);
    }
  }

  destroy(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
  }
}
