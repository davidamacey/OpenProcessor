/**
 * The assist dialog's state: what is in scope, which sample the VLM sees,
 * which actions run and with which endpoint, the server's dry-run plan, and
 * the run once it starts.
 *
 * Thin by design: nothing here sizes a sample, estimates a call count or
 * decides a refusal. Each change re-asks the server for a dry run and the
 * dialog shows its answer (`resolved`) or its refusal verbatim. The sample
 * choice and the action list only choose which request to send.
 */
import { cancelAssistRun, createAssistRun, getAssistRun } from '$lib/api_assist';
import { structuredDetail } from '$lib/api_errors';
import { vlmRunErrorText } from '$lib/vlm/runErrors';
import type {
  AssistAction,
  AssistRunRequest,
  AssistRunState,
  AssistStatus,
  AssistStratifyBy,
  AssistTarget,
} from '$lib/types_assist';

/** What the VLM sees: the user's own selection, the whole scope, or a seeded sample of it. */
export type SampleChoice = 'selected' | 'all' | 'representative';

export interface AssistOpener {
  /** Names what is in scope, for the dialog heading. */
  title: string;
  /** A selection of crops (one crop is a selection of one) or named boxes. */
  selected?: AssistTarget;
  /** A whole cluster, sub-cluster or filter. */
  whole?: AssistTarget;
}

export interface AssistRunDeps {
  create: (body: AssistRunRequest, signal?: AbortSignal) => Promise<AssistRunState>;
  get: (runId: string, signal?: AbortSignal) => Promise<AssistRunState>;
  cancel: (runId: string) => Promise<AssistRunState>;
  debounceMs: number;
  pollMs: number;
}

const DEFAULT_DEPS: AssistRunDeps = {
  create: createAssistRun,
  get: getAssistRun,
  cancel: cancelAssistRun,
  debounceMs: 250,
  pollMs: 1500,
};

const LIVE: AssistStatus[] = ['queued', 'running', 'waiting_region'];

/** True while a started run is still working. */
export function isLiveStatus(status: AssistStatus | undefined): boolean {
  return status !== undefined && LIVE.includes(status);
}

/** Which actions the contract takes for each kind of target; the server still refuses a bad mix. */
const ACTIONS_FOR: Record<AssistTarget['kind'], AssistAction[]> = {
  boxes: ['attributes', 'verify'],
  crops: ['classify', 'attributes', 'verify', 'segment'],
  cluster: ['classify', 'attributes', 'segment'],
  filter: ['classify', 'attributes', 'segment'],
};

export class AssistRunController {
  opener = $state<AssistOpener | null>(null);
  choice = $state<SampleChoice>('all');
  actions = $state<AssistAction[]>(['classify']);
  size = $state<number | null>(null);
  seed = $state(0);
  stratifyBy = $state<AssistStratifyBy>('auto');
  vlm = $state<string | null>(null);
  acknowledgeExternal = $state(false);
  promptPack = $state<string | null>(null);
  segmentVerify = $state<'vlm' | 'none'>('vlm');
  /** The agreement floor of the label proposal; null = the server's default. */
  minAgreement = $state<number | null>(null);

  /** The server's dry run for the current request, or null while unknown. */
  preview = $state<AssistRunState | null>(null);
  planning = $state(false);
  planError = $state<string | null>(null);
  planErrorCode = $state<string | null>(null);

  run = $state<AssistRunState | null>(null);
  running = $state(false);
  startError = $state<string | null>(null);

  #deps: AssistRunDeps;
  #planSeq = 0;
  #planAbort: AbortController | null = null;
  #timer: ReturnType<typeof setTimeout> | undefined;

  constructor(deps: AssistRunDeps = DEFAULT_DEPS) {
    this.#deps = deps;
  }

  get isOpen(): boolean {
    return this.opener !== null;
  }

  get choices(): SampleChoice[] {
    const o = this.opener;
    if (!o) return [];
    return [
      ...(o.selected ? (['selected'] as const) : []),
      ...(o.whole ? (['all', 'representative'] as const) : []),
    ];
  }

  get target(): AssistTarget | null {
    const o = this.opener;
    if (!o) return null;
    return this.choice === 'selected' ? (o.selected ?? null) : (o.whole ?? null);
  }

  get allowedActions(): AssistAction[] {
    const t = this.target;
    return t ? ACTIONS_FOR[t.kind] : [];
  }

  get request(): AssistRunRequest {
    const target = this.target;
    if (!target) throw new Error('assist dialog has no target');
    const req: AssistRunRequest = {
      target,
      sample:
        this.choice === 'representative'
          ? {
              mode: 'representative',
              size: this.size,
              seed: this.seed,
              stratify_by: this.stratifyBy,
            }
          : { mode: 'all' },
      actions: this.actions,
      dry_run: true,
    };
    if (this.vlm) req.vlm = this.vlm;
    if (this.promptPack) req.prompt_pack = this.promptPack;
    if (this.acknowledgeExternal) req.acknowledge_external = true;
    if (this.minAgreement !== null) req.min_agreement = this.minAgreement;
    if (this.actions.includes('segment')) req.segment = { verify: this.segmentVerify };
    return req;
  }

  /** Serialised request, so a view can re-plan whenever it changes. */
  get requestKey(): string {
    return this.target ? JSON.stringify(this.request) : '';
  }

  get canStart(): boolean {
    return (
      this.preview !== null && !this.planning && !this.running && this.planError === null
    );
  }

  open(opener: AssistOpener): void {
    this.#cancelPlan();
    this.opener = opener;
    this.choice = opener.selected ? 'selected' : 'all';
    this.#resetActions();
    this.size = null;
    this.seed = 0;
    this.stratifyBy = 'auto';
    this.minAgreement = null;
    this.preview = null;
    this.planError = null;
    this.planErrorCode = null;
    this.run = null;
    this.running = false;
    this.startError = null;
  }

  close(): void {
    this.#cancelPlan();
    this.opener = null;
  }

  setChoice(next: SampleChoice): void {
    if (!this.choices.includes(next)) return;
    this.choice = next;
    this.#resetActions();
  }

  toggleAction(action: AssistAction): void {
    if (!this.allowedActions.includes(action)) return;
    const on = this.actions.includes(action);
    if (on && this.actions.length === 1) return;
    this.actions = on
      ? this.actions.filter((a) => a !== action)
      : ALL_ACTIONS.filter((a) => a === action || this.actions.includes(a));
    // Boxes the worker writes unverified are judged by this run's own step.
    if (this.actions.includes('segment') && this.actions.includes('verify')) {
      this.segmentVerify = 'none';
    }
  }

  #resetActions(): void {
    const t = this.target;
    this.actions = t?.kind === 'boxes' ? ['verify'] : ['classify'];
    this.segmentVerify = 'vlm';
  }

  /** Re-plan after the debounce; used by the view's effect. */
  schedulePlan(): void {
    clearTimeout(this.#timer);
    this.#timer = setTimeout(() => void this.plan(), this.#deps.debounceMs);
  }

  #cancelPlan(): void {
    clearTimeout(this.#timer);
    this.#planAbort?.abort();
    this.#planSeq += 1;
    this.planning = false;
  }

  async plan(): Promise<void> {
    if (!this.target) return;
    clearTimeout(this.#timer);
    this.#planAbort?.abort();
    const seq = ++this.#planSeq;
    const ctl = new AbortController();
    this.#planAbort = ctl;
    this.planning = true;
    try {
      const res = await this.#deps.create(this.request, ctl.signal);
      if (seq !== this.#planSeq) return;
      this.preview = res;
      this.planError = null;
      this.planErrorCode = null;
    } catch (e) {
      if (seq !== this.#planSeq || (e as Error)?.name === 'AbortError') return;
      this.preview = null;
      this.planError = vlmRunErrorText(e);
      this.planErrorCode =
        structuredDetail<{ error: string; message: string }>(e)?.error ?? null;
    } finally {
      if (seq === this.#planSeq) this.planning = false;
    }
  }

  async start(): Promise<void> {
    if (!this.canStart) return;
    this.running = true;
    this.startError = null;
    try {
      this.run = await this.#deps.create({ ...this.request, dry_run: false });
    } catch (e) {
      this.startError = vlmRunErrorText(e);
      this.running = false;
      return;
    }
    try {
      await this.#follow();
    } catch (e) {
      this.startError = vlmRunErrorText(e);
    } finally {
      this.running = false;
    }
  }

  async #follow(): Promise<void> {
    while (isLiveStatus(this.run?.status)) {
      await new Promise((resolve) => setTimeout(resolve, this.#deps.pollMs));
      const id = this.run?.run_id;
      if (!id || !isLiveStatus(this.run?.status)) return;
      const next = await this.#deps.get(id);
      // A cancel answered while this read was in flight already holds the end state.
      if (!isLiveStatus(this.run?.status)) return;
      this.run = next;
    }
  }

  async cancelRun(): Promise<void> {
    const id = this.run?.run_id;
    if (!id) return;
    try {
      this.run = await this.#deps.cancel(id);
    } catch (e) {
      this.startError = vlmRunErrorText(e);
    }
  }
}

const ALL_ACTIONS: AssistAction[] = ['classify', 'attributes', 'verify', 'segment'];
