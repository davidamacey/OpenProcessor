/**
 * State of the auto-label loop panel: the request form, the server's dry-run
 * plan for it, the newest loop (followed by polling while it is live), its node
 * table, a node's members and the rollback.
 *
 * Thin by design: nothing here sizes a sample, judges a budget or decides what
 * a rollback restores. Each form change asks the server for a dry run and the
 * panel shows its answer, or its refusal verbatim.
 */
import {
  cancelAssistLoop,
  createAssistLoop,
  getAssistLoop,
  listAssistLoops,
  listLoopNodeMembers,
  listLoopNodes,
  resumeAssistLoop,
  rollbackAssistLoop,
} from '$lib/api_assistLoop';
import { ApiError } from '$lib/api_errors';
import { vlmRunErrorText } from '$lib/vlm/runErrors';
import {
  LIVE_LOOP_STATUSES,
  type AssistLoopRequest,
  type AssistLoopState,
  type AssistLoopStatus,
  type LoopNode,
  type LoopRollbackResponse,
} from '$lib/types_assistLoop';

export interface AssistLoopDeps {
  create: typeof createAssistLoop;
  list: typeof listAssistLoops;
  get: typeof getAssistLoop;
  nodes: typeof listLoopNodes;
  members: typeof listLoopNodeMembers;
  cancel: typeof cancelAssistLoop;
  resume: typeof resumeAssistLoop;
  rollback: typeof rollbackAssistLoop;
  debounceMs: number;
  pollMs: number;
}

const DEFAULT_DEPS: AssistLoopDeps = {
  create: createAssistLoop,
  list: listAssistLoops,
  get: getAssistLoop,
  nodes: listLoopNodes,
  members: listLoopNodeMembers,
  cancel: cancelAssistLoop,
  resume: resumeAssistLoop,
  rollback: rollbackAssistLoop,
  debounceMs: 300,
  pollMs: 2000,
};

const NODE_PAGE = 100;

/** True while the worker owns the loop (it holds cluster training meanwhile). */
export function isLiveLoop(status: AssistLoopStatus | undefined): boolean {
  return status !== undefined && LIVE_LOOP_STATUSES.includes(status);
}

/** A loop that stopped and the server will queue again. */
export function isResumableLoop(status: AssistLoopStatus | undefined): boolean {
  return status === 'paused' || status === 'interrupted' || status === 'cancelled';
}

export class AssistLoopController {
  // The form. Numbers the server bounds are sent as typed; its 422 is shown.
  minRootSize = $state(40);
  maxVlmCrops = $state<number | null>(2000);
  theta = $state(0.9);
  maxDepth = $state(3);
  seed = $state(0);
  vlm = $state<string | null>(null);
  acknowledgeExternal = $state(false);
  auditVlm = $state<string | null>(null);
  auditAcknowledgeExternal = $state(false);
  auditOnFail = $state<'pause' | 'rollback'>('pause');

  /** False when the backend does not serve the loop routes: the panel is absent. */
  supported = $state<boolean | null>(null);
  loadError = $state<string | null>(null);

  preview = $state<AssistLoopState | null>(null);
  planning = $state(false);
  planError = $state<string | null>(null);

  loop = $state<AssistLoopState | null>(null);
  actionError = $state<string | null>(null);
  busy = $state(false);

  nodes = $state<LoopNode[]>([]);
  nodesNext = $state<string | null>(null);
  nodesLoading = $state(false);
  decisionFilter = $state<string>('');
  nodesError = $state<string | null>(null);

  openNode = $state<string | null>(null);
  members = $state<string[]>([]);
  membersNext = $state<string | null>(null);
  membersError = $state<string | null>(null);

  rollbackPlan = $state<LoopRollbackResponse | null>(null);
  rollbackResult = $state<LoopRollbackResponse | null>(null);
  rollbackNodes = $state<string[] | null>(null);

  #deps: AssistLoopDeps;
  #planSeq = 0;
  #planAbort: AbortController | null = null;
  #planTimer: ReturnType<typeof setTimeout> | undefined;
  #pollTimer: ReturnType<typeof setTimeout> | undefined;
  #disposed = false;

  constructor(deps: AssistLoopDeps = DEFAULT_DEPS) {
    this.#deps = deps;
  }

  get live(): boolean {
    return isLiveLoop(this.loop?.status);
  }

  get request(): AssistLoopRequest {
    return {
      scope: { cluster_ids: null, min_root_size: this.minRootSize },
      vlm: this.vlm,
      prompt_pack: null,
      acknowledge_external: this.acknowledgeExternal,
      seed: this.seed,
      gate: { theta: this.theta, alpha: 0.05, max_dissent: 1 },
      recursion: { max_depth: this.maxDepth, branching: 4, min_node_size: 40 },
      budget:
        this.maxVlmCrops === null
          ? null
          : { max_vlm_crops: this.maxVlmCrops, max_vlm_requests: null, max_wall_s: null },
      audit: {
        vlm: this.auditVlm,
        acknowledge_external: this.auditAcknowledgeExternal,
        rate: 0.01,
        min_per_batch: 30,
        max_crops: 500,
        floor: 0.85,
        on_fail: this.auditOnFail,
      },
      vlm_concurrency: 16,
      node_concurrency: 8,
      dry_run: true,
    };
  }

  /** Serialised request, so a view can re-plan whenever the form changes. */
  get requestKey(): string {
    return JSON.stringify(this.request);
  }

  get canStart(): boolean {
    return (
      this.preview !== null &&
      this.maxVlmCrops !== null &&
      !this.planning &&
      !this.busy &&
      this.planError === null &&
      !this.live
    );
  }

  /** Read the newest loop of the project, if any, and follow it while it is live. */
  async load(): Promise<void> {
    try {
      const res = await this.#deps.list({ limit: 1 });
      this.supported = true;
      this.loadError = null;
      const newest = res.loops[0] ?? null;
      if (newest) await this.#adopt(newest);
    } catch (e) {
      if (e instanceof ApiError && (e.status === 404 || e.status === 501)) {
        this.supported = false;
        return;
      }
      this.supported = true;
      this.loadError = vlmRunErrorText(e);
    }
  }

  dispose(): void {
    this.#disposed = true;
    clearTimeout(this.#planTimer);
    clearTimeout(this.#pollTimer);
    this.#planAbort?.abort();
  }

  schedulePlan(): void {
    clearTimeout(this.#planTimer);
    this.#planTimer = setTimeout(() => void this.plan(), this.#deps.debounceMs);
  }

  async plan(): Promise<void> {
    clearTimeout(this.#planTimer);
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
    } catch (e) {
      if (seq !== this.#planSeq || (e as Error)?.name === 'AbortError') return;
      this.preview = null;
      this.planError = vlmRunErrorText(e);
    } finally {
      if (seq === this.#planSeq) this.planning = false;
    }
  }

  async start(): Promise<void> {
    if (!this.canStart) return;
    this.busy = true;
    this.actionError = null;
    try {
      const started = await this.#deps.create({ ...this.request, dry_run: false });
      await this.#adopt(started);
    } catch (e) {
      this.actionError = vlmRunErrorText(e);
    } finally {
      this.busy = false;
    }
  }

  async cancel(): Promise<void> {
    await this.#act((id) => this.#deps.cancel(id));
  }

  async resume(): Promise<void> {
    await this.#act((id) => this.#deps.resume(id));
  }

  async #act(call: (loopId: string) => Promise<AssistLoopState>): Promise<void> {
    const id = this.loop?.loop_id;
    if (!id) return;
    this.busy = true;
    this.actionError = null;
    try {
      await this.#adopt(await call(id));
    } catch (e) {
      this.actionError = vlmRunErrorText(e);
    } finally {
      this.busy = false;
    }
  }

  async #adopt(state: AssistLoopState): Promise<void> {
    this.loop = state;
    await this.loadNodes(true);
    this.#schedulePoll();
  }

  #schedulePoll(): void {
    clearTimeout(this.#pollTimer);
    if (this.#disposed || !this.live) return;
    this.#pollTimer = setTimeout(() => void this.#poll(), this.#deps.pollMs);
  }

  async #poll(): Promise<void> {
    const id = this.loop?.loop_id;
    if (!id || !this.live) return;
    try {
      const next = await this.#deps.get(id);
      if (this.#disposed) return;
      this.loop = next;
      await this.loadNodes(true);
    } catch (e) {
      this.actionError = vlmRunErrorText(e);
    }
    this.#schedulePoll();
  }

  async loadNodes(reset: boolean): Promise<void> {
    const id = this.loop?.loop_id;
    if (!id) return;
    this.nodesLoading = true;
    try {
      // A reset re-reads from the top as many pages as are shown, so a live
      // table updates in place instead of collapsing to its first page.
      const want = reset ? Math.max(this.nodes.length, NODE_PAGE) : NODE_PAGE;
      const page = await this.#deps.nodes(id, {
        decision: this.decisionFilter || undefined,
        after: reset ? undefined : this.nodesNext,
        limit: Math.min(500, want),
      });
      this.nodes = reset ? page.nodes : [...this.nodes, ...page.nodes];
      this.nodesNext = page.next_after;
      this.nodesError = null;
    } catch (e) {
      this.nodesError = vlmRunErrorText(e);
    } finally {
      this.nodesLoading = false;
    }
  }

  async setDecisionFilter(next: string): Promise<void> {
    this.decisionFilter = next;
    this.nodes = [];
    await this.loadNodes(true);
  }

  async toggleNode(path: string): Promise<void> {
    if (this.openNode === path) {
      this.openNode = null;
      return;
    }
    this.openNode = path;
    this.members = [];
    this.membersNext = null;
    this.membersError = null;
    await this.moreMembers();
  }

  async moreMembers(): Promise<void> {
    const id = this.loop?.loop_id;
    const path = this.openNode;
    if (!id || !path) return;
    try {
      const page = await this.#deps.members(id, path, {
        after: this.membersNext,
        limit: 200,
      });
      if (this.openNode !== path) return;
      this.members = [...this.members, ...page.crop_ids];
      this.membersNext = page.next_after;
    } catch (e) {
      this.membersError = vlmRunErrorText(e);
    }
  }

  /** Ask what a rollback would restore (nothing is written). `nodePaths` null = every node. */
  async previewRollback(nodePaths: string[] | null): Promise<void> {
    const id = this.loop?.loop_id;
    if (!id) return;
    this.busy = true;
    this.actionError = null;
    this.rollbackResult = null;
    try {
      this.rollbackPlan = await this.#deps.rollback(id, {
        node_paths: nodePaths,
        dry_run: true,
      });
      this.rollbackNodes = nodePaths;
    } catch (e) {
      this.rollbackPlan = null;
      this.actionError = vlmRunErrorText(e);
    } finally {
      this.busy = false;
    }
  }

  async confirmRollback(): Promise<void> {
    const id = this.loop?.loop_id;
    if (!id || !this.rollbackPlan) return;
    this.busy = true;
    this.actionError = null;
    try {
      this.rollbackResult = await this.#deps.rollback(id, {
        node_paths: this.rollbackNodes,
        dry_run: false,
      });
      this.rollbackPlan = null;
      this.loop = await this.#deps.get(id);
      await this.loadNodes(true);
    } catch (e) {
      this.actionError = vlmRunErrorText(e);
    } finally {
      this.busy = false;
    }
  }

  dismissRollback(): void {
    this.rollbackPlan = null;
  }
}
