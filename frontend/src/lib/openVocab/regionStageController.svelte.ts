/**
 * The region stage's served state (`GET /region_stage`) and its pause and
 * resume writes. A write is only ever sent after the operator confirmed it
 * (`ask` then `confirm`); the served state in the response replaces what is
 * on screen. The "re-run gate-skipped" request is the served
 * `rerun_skipped`, handed to the reprocess dialog as served.
 */
import { apiErrorText } from '$lib/api';
import {
  getRegionStage,
  pauseRegionStage,
  resumeRegionStage,
} from '$lib/api_regionStage';
import type { ReprocessRequest } from '$lib/types_import';
import type { RegionStageState } from '$lib/types_openVocab';

export interface RegionStageDeps {
  getRegionStage: typeof getRegionStage;
  pauseRegionStage: typeof pauseRegionStage;
  resumeRegionStage: typeof resumeRegionStage;
}

export class RegionStage {
  state = $state<RegionStageState | null>(null);
  loadError = $state<string | null>(null);
  confirming = $state<'pause' | 'resume' | null>(null);
  busy = $state(false);
  actionError = $state<string | null>(null);

  #deps: RegionStageDeps;

  constructor(deps: Partial<RegionStageDeps> = {}) {
    this.#deps = { getRegionStage, pauseRegionStage, resumeRegionStage, ...deps };
  }

  get rerunRequest(): ReprocessRequest | null {
    return this.state?.rerun_skipped ?? null;
  }

  async load(): Promise<void> {
    try {
      this.state = await this.#deps.getRegionStage();
      this.loadError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = apiErrorText(e);
    }
  }

  ask(kind: 'pause' | 'resume'): void {
    this.actionError = null;
    this.confirming = kind;
  }

  cancel(): void {
    this.confirming = null;
    this.actionError = null;
  }

  async confirm(): Promise<boolean> {
    const kind = this.confirming;
    if (!kind || this.busy) return false;
    this.busy = true;
    this.actionError = null;
    try {
      this.state = await (kind === 'pause'
        ? this.#deps.pauseRegionStage()
        : this.#deps.resumeRegionStage());
      this.confirming = null;
      return true;
    } catch (e) {
      this.actionError = apiErrorText(e);
      return false;
    } finally {
      this.busy = false;
    }
  }
}

export function createRegionStage(deps: Partial<RegionStageDeps> = {}): RegionStage {
  return new RegionStage(deps);
}
