/**
 * Applying a run's cluster proposal: the served dry-run count first, then
 * the explicit write. The server decides what is applicable (locked,
 * validated and holdout members are left alone), whether the scope still
 * matches the run (`scope_changed`) and whether the class exists
 * (`class_not_in_registry`); this controller shows those answers.
 */
import { applyAssistRun } from '$lib/api_assist';
import { apiErrorText, structuredDetail } from '$lib/api_errors';
import type { AssistApplyRequest, AssistApplyResponse } from '$lib/types_assist';

export interface ProposalApplyDeps {
  apply: (
    runId: string,
    body: AssistApplyRequest,
    signal?: AbortSignal,
  ) => Promise<AssistApplyResponse>;
}

const DEFAULT_DEPS: ProposalApplyDeps = { apply: applyAssistRun };

export class ProposalApplyController {
  className = $state<string | null>(null);
  dry = $state<AssistApplyResponse | null>(null);
  applied = $state<AssistApplyResponse | null>(null);
  busy = $state(false);
  error = $state<string | null>(null);
  errorCode = $state<string | null>(null);

  #runId: string;
  #deps: ProposalApplyDeps;
  #onapplied: (() => void) | undefined;
  #seq = 0;
  /** The caller overrode the refusal to apply to a changed scope; the write repeats it. */
  #allowChanged = false;

  constructor(
    runId: string,
    deps: ProposalApplyDeps = DEFAULT_DEPS,
    onapplied?: () => void,
  ) {
    this.#runId = runId;
    this.#deps = deps;
    this.#onapplied = onapplied;
  }

  choose(name: string | null): void {
    this.className = name;
    this.dry = null;
    this.applied = null;
    this.error = null;
    this.errorCode = null;
    this.#allowChanged = false;
    this.#seq += 1;
  }

  #fail(e: unknown): void {
    this.error = apiErrorText(e);
    this.errorCode =
      structuredDetail<{ error: string; message: string }>(e)?.error ?? null;
  }

  async preview(opts: { allowChangedScope?: boolean } = {}): Promise<void> {
    if (!this.className) return;
    const seq = ++this.#seq;
    this.#allowChanged = opts.allowChangedScope === true;
    this.busy = true;
    this.error = null;
    this.errorCode = null;
    try {
      const res = await this.#deps.apply(this.#runId, this.#body(true));
      if (seq === this.#seq) this.dry = res;
    } catch (e) {
      if (seq !== this.#seq) return;
      this.dry = null;
      this.#fail(e);
    } finally {
      if (seq === this.#seq) this.busy = false;
    }
  }

  async apply(): Promise<void> {
    if (!this.className || this.busy) return;
    this.busy = true;
    this.error = null;
    this.errorCode = null;
    try {
      this.applied = await this.#deps.apply(this.#runId, this.#body(false));
      this.dry = null;
      this.#onapplied?.();
    } catch (e) {
      this.#fail(e);
    } finally {
      this.busy = false;
    }
  }

  #body(dryRun: boolean): AssistApplyRequest {
    const body: AssistApplyRequest = {
      class_name: this.className!,
      label_source: 'human',
      dry_run: dryRun,
    };
    if (this.#allowChanged) body.allow_changed_scope = true;
    return body;
  }
}
