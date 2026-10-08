/**
 * What is active on one config axis (`GET /{resource}/active`) and the
 * writes that move it: activate, rollback and, where the resource has one,
 * deactivate (any_domain_plan.md §4.4, §7.2, §7.3, §7.6 item 4). Shared by
 * the prompt-pack (W3) and region-profile (W4) editors.
 *
 * Every write sends `expected_active` = the `active` ref of the last read,
 * exactly as served. A 409 `active_conflict` re-reads, so the next attempt
 * starts from what is really active. A 422 `validation_failed` keeps the
 * served report; `force` is only ever sent by a caller that saw its
 * `force_allowed`.
 */
import { configErrorDetail, apiErrorText } from '$lib/api';
import type {
  ActiveConfigResponse,
  ActiveRef,
  ConfigActivateRequest,
  ConfigErrorDetail,
  ValidationReport,
} from '$lib/types_config';

/** The resource's routes. `W` is the activate response (a profile's
 *  carries `impact` and `validation`, §7.3). */
export interface ConfigActiveBackend<
  W extends ActiveConfigResponse = ActiveConfigResponse,
> {
  getActive: () => Promise<ActiveConfigResponse>;
  activate: (
    name: string,
    body: ConfigActivateRequest & Record<string, unknown>,
  ) => Promise<W>;
  rollback: (body: { expected_active: ActiveRef }) => Promise<ActiveConfigResponse>;
  /** Absent for a resource without a deactivate route (packs). */
  deactivate?: (body: { expected_active: ActiveRef }) => Promise<ActiveConfigResponse>;
}

export class ConfigActive<W extends ActiveConfigResponse = ActiveConfigResponse> {
  active = $state<ActiveConfigResponse | null>(null);
  loadError = $state<string | null>(null);
  busy = $state(false);
  actionError = $state<string | null>(null);
  /** The served report of a refused activation (422 `validation_failed`). */
  activateReport = $state<ValidationReport | null>(null);
  /** The served detail of the last refused write (a VLM activation reads
   *  `vlm_external_not_acknowledged` off it); null after a success. */
  errorDetail = $state<ConfigErrorDetail | null>(null);
  /** The served response of the last successful activation. */
  lastActivation = $state<W | null>(null);
  /** Runs after every successful write (a profile editor re-polls
   *  `/health` so the app's "reload to apply" notice can fire). */
  onchanged: (() => void) | null = null;

  #backend: ConfigActiveBackend<W>;

  constructor(backend: ConfigActiveBackend<W>) {
    this.#backend = backend;
  }

  /** True when this resource has a deactivate route. */
  get deactivatable(): boolean {
    return this.#backend.deactivate != null;
  }

  async load(): Promise<void> {
    try {
      this.active = await this.#backend.getActive();
      this.loadError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = apiErrorText(e);
    }
  }

  clearAction(): void {
    this.actionError = null;
    this.activateReport = null;
    this.errorDetail = null;
  }

  /** Re-activates the served `previous`. */
  async rollback(): Promise<boolean> {
    return this.#write((expected) =>
      this.#backend.rollback({ expected_active: expected }),
    );
  }

  /** Turns the axis off (a region profile: region detection stops). */
  async deactivate(): Promise<boolean> {
    const deactivate = this.#backend.deactivate;
    if (!deactivate) return false;
    return this.#write((expected) => deactivate({ expected_active: expected }));
  }

  /** Makes `name@revision` the active one. `extra` is spread into the
   *  body (a VLM activation's `acknowledge_external`); packs and profiles
   *  pass nothing. */
  async activate(
    name: string,
    revision: number | null,
    force: boolean,
    extra?: Record<string, unknown>,
  ): Promise<boolean> {
    return this.#write(async (expected) => {
      const res = await this.#backend.activate(name, {
        revision,
        expected_active: expected,
        force,
        ...extra,
      });
      this.lastActivation = res;
      return res;
    });
  }

  async #write(
    call: (expected: ActiveRef) => Promise<ActiveConfigResponse>,
  ): Promise<boolean> {
    const current = this.active;
    if (!current || this.busy) return false;
    this.busy = true;
    this.clearAction();
    this.lastActivation = null;
    try {
      this.active = await call(current.active);
      this.onchanged?.();
      return true;
    } catch (e) {
      this.actionError = apiErrorText(e);
      const d = configErrorDetail(e);
      this.errorDetail = d;
      if (d?.report) this.activateReport = d.report;
      if (d?.error === 'active_conflict') await this.load();
      return false;
    } finally {
      this.busy = false;
    }
  }
}
