/**
 * The state and run loop both test-on-crop controllers share (prompt packs, region
 * profiles): the draft-or-saved source with a forced-saved override, the optional VLM pick,
 * the in-flight run that a rerun aborts, and the served error/report/missing-crop refusal.
 * A subclass adds its own inputs, `ready` and `request`.
 */
import { untrack } from 'svelte';
import { apiErrorText } from '$lib/api_errors';
import { configErrorDetail } from '$lib/api_profiles';
import type { TestVlmSelection } from '$lib/types_configTest';
import type { ValidationReport } from '$lib/types_config';

export type TestSource = 'draft' | 'saved';

export abstract class ConfigTestRun<Ctx, Req, Res> {
  source = $state<TestSource>('draft');
  /** The VLM to answer, when the operator picked one; null = the server's
   *  default (the active endpoint). */
  vlmSelection = $state<TestVlmSelection | null>(null);

  running = $state(false);
  result = $state<Res | null>(null);
  error = $state<string | null>(null);
  errorReport = $state<ValidationReport | null>(null);
  /** Ids the server named in a `crop_not_found` refusal. */
  missingCropIds = $state<string[]>([]);

  #test: (req: Req, signal: AbortSignal) => Promise<Res>;
  #abort: AbortController | null = null;
  #forcedFrom: TestSource | null = null;

  constructor(test: (req: Req, signal: AbortSignal) => Promise<Res>) {
    this.#test = test;
  }

  /** Enough input to run (a rerun while one is in flight aborts it). */
  abstract get ready(): boolean;

  abstract request(ctx: Ctx): Req;

  /** What the Run button enables on. */
  get canRun(): boolean {
    return !this.running && this.ready;
  }

  /** No draft to test (a read-only document, or a revision being viewed):
   *  forces the saved source; when a draft is back, restores what the
   *  operator had before, never overriding a choice they made. */
  setSavedOnly(savedOnly: boolean): void {
    if (savedOnly && this.#forcedFrom == null) {
      this.#forcedFrom = untrack(() => this.source);
      this.source = 'saved';
    } else if (!savedOnly && this.#forcedFrom != null) {
      this.source = this.#forcedFrom;
      this.#forcedFrom = null;
    }
  }

  async run(ctx: Ctx): Promise<void> {
    if (!this.ready) return;
    this.#abort?.abort();
    const ctl = new AbortController();
    this.#abort = ctl;
    this.running = true;
    this.error = null;
    this.errorReport = null;
    this.missingCropIds = [];
    try {
      this.result = await this.#test(this.request(ctx), ctl.signal);
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.result = null;
      this.error = apiErrorText(e);
      const d = configErrorDetail(e);
      this.errorReport = d?.report ?? null;
      this.missingCropIds = d?.error === 'crop_not_found' ? (d.crop_ids ?? []) : [];
    } finally {
      if (this.#abort === ctl) {
        this.running = false;
        this.#abort = null;
      }
    }
  }

  stop(): void {
    this.#abort?.abort();
  }
}
