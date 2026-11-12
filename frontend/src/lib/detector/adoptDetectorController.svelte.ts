/**
 * The "Adopt detector classes" dialog: a dry run first (what would be created, how many
 * items, what is blocked), then the confirmed write, followed to its end. Every count is the
 * server's; the controller only sequences the two calls.
 */
import { adoptDetector } from '$lib/api_labelOps';
import { runLabelOp, type LabelOpOutcome } from '$lib/labelOps/runLabelOp';
import type { AdoptDetectorResponse } from '$lib/types_labelOps';

export type AdoptPhase = 'idle' | 'checking' | 'ready' | 'writing' | 'done' | 'failed';

export class AdoptDetectorController {
  phase = $state<AdoptPhase>('idle');
  preview = $state<AdoptDetectorResponse | null>(null);
  /** The confirmed write's answer: `remaining` and the `op_id` to undo by. */
  written = $state<AdoptDetectorResponse | null>(null);
  outcome = $state<LabelOpOutcome | null>(null);
  failure = $state<unknown>(null);

  #adopt: typeof adoptDetector;
  #run: typeof runLabelOp;

  constructor(adopt = adoptDetector, run = runLabelOp) {
    this.#adopt = adopt;
    this.#run = run;
  }

  async check(): Promise<void> {
    this.phase = 'checking';
    this.failure = null;
    try {
      this.preview = await this.#adopt(true);
      this.phase = 'ready';
    } catch (e) {
      this.failure = e;
      this.phase = 'failed';
    }
  }

  async confirm(): Promise<void> {
    this.phase = 'writing';
    this.failure = null;
    try {
      const res = await this.#adopt(false);
      this.written = res;
      this.outcome = await this.#run({
        op_id: res.op_id,
        job_id: res.job_id,
        status: null,
      });
      this.phase = 'done';
    } catch (e) {
      this.failure = e;
      this.phase = 'failed';
    }
  }
}
