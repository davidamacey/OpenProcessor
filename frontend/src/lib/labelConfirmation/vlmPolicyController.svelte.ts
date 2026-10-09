/**
 * The VLM scope panel's state (`GET/PUT /vlm/policy`), #119.
 *
 * Thin: nothing here validates a knob (the server's 422 is shown as
 * served), estimates a cost or decides what the VLM will label. The draft
 * is the served policy minus `revision`; a save sends it with
 * `expected_revision` = the revision last read, and a 409
 * `revision_conflict` offers Reload (drop edits) or Keep my edits.
 */
import { detectorErrorDetail, detectorErrorLines } from '$lib/api_detector';
import { getVlmPolicy, putVlmPolicy } from '$lib/api_labelConfirmation';
import type {
  VlmPolicy,
  VlmPolicyBody,
  VlmPolicyUpdate,
} from '$lib/types_labelConfirmation';

export interface VlmPolicyDeps {
  getPolicy: (signal?: AbortSignal) => Promise<VlmPolicy>;
  putPolicy: (req: VlmPolicyUpdate, signal?: AbortSignal) => Promise<VlmPolicy>;
}

const DEFAULT_DEPS: VlmPolicyDeps = { getPolicy: getVlmPolicy, putPolicy: putVlmPolicy };

function bodyOf(p: VlmPolicy): VlmPolicyBody {
  const { revision: _revision, ...body } = p;
  void _revision;
  return structuredClone(body);
}

export class VlmPolicyEditor {
  loading = $state(true);
  loadError = $state<string | null>(null);
  revision = $state<number | null>(null);
  draft = $state<VlmPolicyBody>({});
  saving = $state(false);
  saveLines = $state<string[]>([]);
  /** True after a successful save, until the next edit. */
  saved = $state(false);
  conflict = $state(false);

  #baseline = $state('');
  #deps: VlmPolicyDeps;

  constructor(deps: VlmPolicyDeps = DEFAULT_DEPS) {
    this.#deps = deps;
  }

  get dirty(): boolean {
    return JSON.stringify(this.draft) !== this.#baseline;
  }

  #adopt(p: VlmPolicy): void {
    this.draft = bodyOf(p);
    this.#baseline = JSON.stringify(this.draft);
    this.revision = p.revision ?? null;
  }

  /** Note an edit: clears the "saved" confirmation. */
  touched(): void {
    this.saved = false;
  }

  async load(signal?: AbortSignal): Promise<void> {
    this.loading = true;
    this.loadError = null;
    try {
      this.#adopt(await this.#deps.getPolicy(signal));
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = detectorErrorLines(e).join(' ');
    } finally {
      this.loading = false;
    }
  }

  async reload(): Promise<void> {
    this.conflict = false;
    this.saveLines = [];
    this.saved = false;
    try {
      this.#adopt(await this.#deps.getPolicy());
    } catch (e) {
      this.saveLines = detectorErrorLines(e);
    }
  }

  /** After a 409: adopt the fresh revision, keep the draft as edited. */
  async keepMyEdits(): Promise<void> {
    this.conflict = false;
    this.saveLines = [];
    try {
      const fresh = await this.#deps.getPolicy();
      this.revision = fresh.revision ?? null;
    } catch (e) {
      this.saveLines = detectorErrorLines(e);
    }
  }

  async save(): Promise<boolean> {
    if (this.revision == null) return false;
    this.saving = true;
    this.saveLines = [];
    this.conflict = false;
    this.saved = false;
    try {
      const res = await this.#deps.putPolicy({
        ...$state.snapshot(this.draft),
        expected_revision: this.revision,
      });
      this.#adopt(res);
      this.saved = true;
      return true;
    } catch (e) {
      this.saveLines = detectorErrorLines(e);
      this.conflict = detectorErrorDetail(e)?.error === 'revision_conflict';
      return false;
    } finally {
      this.saving = false;
    }
  }
}
