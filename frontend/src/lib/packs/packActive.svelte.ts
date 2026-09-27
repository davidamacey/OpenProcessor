/**
 * The active prompt pack of the current project (`GET /prompt_packs/active`)
 * and the two writes that move it: activate and rollback
 * (any_domain_plan.md §7.2, §7.6 item 4; docs/design/
 * w3-pack-editor-ui-plan-2026-09-27.md §2, §3).
 *
 * Both writes send `expected_active` = the `active` ref of the last read,
 * exactly as served. A 409 `active_conflict` re-reads, so the next attempt
 * starts from what is really active. A 422 `validation_failed` keeps the
 * served report; `force` is only ever sent by a caller that saw its
 * `force_allowed`.
 */
import {
  activatePromptPack,
  getActivePromptPack,
  packErrorDetail,
  packErrorText,
  rollbackPromptPack,
} from '$lib/api';
import type { ActiveConfigResponse, ValidationReport } from '$lib/types_packs';

export interface PackActiveDeps {
  getActivePromptPack: typeof getActivePromptPack;
  activatePromptPack: typeof activatePromptPack;
  rollbackPromptPack: typeof rollbackPromptPack;
}

const DEFAULT_DEPS: PackActiveDeps = {
  getActivePromptPack,
  activatePromptPack,
  rollbackPromptPack,
};

export class PackActive {
  active = $state<ActiveConfigResponse | null>(null);
  loadError = $state<string | null>(null);
  busy = $state(false);
  actionError = $state<string | null>(null);
  /** The served report of a refused activation (422 `validation_failed`). */
  activateReport = $state<ValidationReport | null>(null);

  #deps: PackActiveDeps;

  constructor(deps: Partial<PackActiveDeps> = {}) {
    this.#deps = { ...DEFAULT_DEPS, ...deps };
  }

  async load(): Promise<void> {
    try {
      this.active = await this.#deps.getActivePromptPack();
      this.loadError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = packErrorText(e);
    }
  }

  clearAction(): void {
    this.actionError = null;
    this.activateReport = null;
  }

  /** Re-activates the served `previous`. */
  async rollback(): Promise<boolean> {
    const current = this.active;
    if (!current || this.busy) return false;
    this.busy = true;
    this.clearAction();
    try {
      this.active = await this.#deps.rollbackPromptPack({
        expected_active: current.active,
      });
      return true;
    } catch (e) {
      await this.#refused(e);
      return false;
    } finally {
      this.busy = false;
    }
  }

  /** Makes `name@revision` the active pack. */
  async activate(
    name: string,
    revision: number | null,
    force: boolean,
  ): Promise<boolean> {
    const current = this.active;
    if (!current || this.busy) return false;
    this.busy = true;
    this.clearAction();
    try {
      this.active = await this.#deps.activatePromptPack(name, {
        revision,
        expected_active: current.active,
        force,
      });
      return true;
    } catch (e) {
      await this.#refused(e);
      return false;
    } finally {
      this.busy = false;
    }
  }

  async #refused(e: unknown): Promise<void> {
    this.actionError = packErrorText(e);
    const d = packErrorDetail(e);
    if (d?.report) this.activateReport = d.report;
    if (d?.error === 'active_conflict') await this.load();
  }
}
