/**
 * NextStepsStore: the served curation loop (`GET {P}/next_steps`). It holds
 * the steps, the first undone one and the view the Workspace opens on; a
 * failed read leaves them empty and the Workspace falls back to Gallery.
 */
import { getNextSteps } from '$lib/api_workflow';
import { onProjectChange } from '$lib/projectChange';
import type { WorkflowStep, WorkflowSteps } from '$lib/types_workflow';

class NextStepsStore {
  steps = $state<WorkflowStep[]>([]);
  next = $state<WorkflowStep | null>(null);
  defaultView = $state<WorkflowSteps['workspace_default_view'] | null>(null);
  loaded = $state(false);
  /** True from the moment the Workspace is entered until the read for that visit has landed:
   *  the cached default view is then an earlier visit's, and must not open the Workspace. */
  stale = $state(false);
  #inflight: Promise<void> | null = null;
  #gen = 0;
  #seq = 0;

  async init(): Promise<void> {
    if (this.loaded) return;
    if (this.#inflight) return this.#inflight;
    return this.refresh();
  }

  /** The Workspace opens: the cached default view is stale until this visit's read lands. */
  enter(): Promise<void> {
    this.stale = true;
    return this.refresh();
  }

  /** Re-read after a write that can finish a step (a label, an activation). */
  refresh(): Promise<void> {
    const gen = this.#gen;
    const seq = ++this.#seq;
    const run = (async () => {
      try {
        const res = await getNextSteps();
        if (gen !== this.#gen) return;
        this.steps = res.steps;
        this.next = res.next;
        this.defaultView = res.workspace_default_view;
      } catch {
        if (gen !== this.#gen) return;
        this.steps = [];
        this.next = null;
        this.defaultView = null;
      } finally {
        if (gen === this.#gen) {
          this.loaded = true;
          if (seq === this.#seq) this.stale = false;
          this.#inflight = null;
        }
      }
    })();
    this.#inflight = run;
    return run;
  }

  reset(): void {
    this.#gen += 1;
    this.#inflight = null;
    this.steps = [];
    this.next = null;
    this.defaultView = null;
    this.loaded = false;
    this.stale = false;
  }
}

export const nextStepsStore = new NextStepsStore();
onProjectChange(() => nextStepsStore.reset());
