/**
 * `/settings/ingest-policy`: edit the project's ingest policy
 * (`GET/PUT /ingest/policy`) with a live cost preview
 * (`POST /ingest/policy/preview`).
 *
 * Thin: nothing here validates the draft (a `selected` mode with no
 * criterion is the server's 422), computes a cost or decides a policy. The
 * draft is the served policy minus `revision`; a save sends it with
 * `expected_revision` = the revision last read; every refusal is shown as
 * served.
 */
import { detectorErrorDetail, detectorErrorLines } from '$lib/api_detector';
import { getIngestPolicy, previewIngestPolicy, putIngestPolicy } from '$lib/api_detector';
import { getIngestConfig } from '$lib/api';
import type {
  IngestDetectorInfo,
  IngestPolicy,
  IngestPolicyBody,
  IngestPolicyPreview,
  IngestPolicyPutResponse,
  IngestPolicyUpdate,
} from '$lib/types_detector';

export interface IngestPolicyDeps {
  getPolicy: (signal?: AbortSignal) => Promise<IngestPolicy>;
  getConfig: (signal?: AbortSignal) => Promise<{ detector: IngestDetectorInfo | null }>;
  putPolicy: (
    req: IngestPolicyUpdate,
    signal?: AbortSignal,
  ) => Promise<IngestPolicyPutResponse>;
  previewPolicy: (
    body: IngestPolicyBody,
    signal?: AbortSignal,
  ) => Promise<IngestPolicyPreview>;
  debounceMs: number;
}

const DEFAULT_DEPS: IngestPolicyDeps = {
  getPolicy: getIngestPolicy,
  getConfig: getIngestConfig,
  putPolicy: putIngestPolicy,
  previewPolicy: previewIngestPolicy,
  debounceMs: 400,
};

function bodyOf(p: IngestPolicy | IngestPolicyPutResponse): IngestPolicyBody {
  const body: IngestPolicyBody = {};
  if (p.detect !== undefined) body.detect = structuredClone(p.detect);
  if (p.embedding !== undefined) body.embedding = structuredClone(p.embedding);
  if (p.detector !== undefined) body.detector = structuredClone(p.detector);
  return body;
}

export class IngestPolicyEditor {
  loading = $state(true);
  loadError = $state<string | null>(null);
  /** The served revision the draft is based on (null until read). */
  revision = $state<number | null>(null);
  draft = $state<IngestPolicyBody>({});
  detector = $state<IngestDetectorInfo | null>(null);

  preview = $state<IngestPolicyPreview | null>(null);
  previewing = $state(false);
  previewError = $state<string | null>(null);

  saving = $state(false);
  saveLines = $state<string[]>([]);
  /** A 409 `revision_conflict`: Reload or Keep my edits. */
  conflict = $state(false);
  unknownNames = $state<string[]>([]);

  #baseline = $state('');
  #deps: IngestPolicyDeps;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #abort: AbortController | null = null;

  constructor(deps: IngestPolicyDeps = DEFAULT_DEPS) {
    this.#deps = deps;
  }

  get labelNames(): string[] {
    return this.detector?.labels.map((l) => l.name) ?? [];
  }

  get dirty(): boolean {
    return JSON.stringify(this.draft) !== this.#baseline;
  }

  #adopt(p: IngestPolicy | IngestPolicyPutResponse): void {
    this.draft = bodyOf(p);
    this.#baseline = JSON.stringify(this.draft);
    this.revision = p.revision ?? null;
  }

  async load(signal?: AbortSignal): Promise<void> {
    this.loading = true;
    this.loadError = null;
    try {
      const [policy, config] = await Promise.all([
        this.#deps.getPolicy(signal),
        this.#deps.getConfig(signal),
      ]);
      this.#adopt(policy);
      this.detector = config.detector;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = detectorErrorLines(e).join(' ');
    } finally {
      this.loading = false;
    }
  }

  /** Re-read the policy and drop every edit. */
  async reload(): Promise<void> {
    this.conflict = false;
    this.saveLines = [];
    this.unknownNames = [];
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

  /** Called on every draft change; one preview per quiet period, the
   *  previous request aborted. */
  schedulePreview(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = setTimeout(() => void this.#runPreview(), this.#deps.debounceMs);
  }

  async #runPreview(): Promise<void> {
    this.#abort?.abort();
    const ctl = new AbortController();
    this.#abort = ctl;
    this.previewing = true;
    this.previewError = null;
    try {
      const res = await this.#deps.previewPolicy($state.snapshot(this.draft), ctl.signal);
      if (ctl.signal.aborted) return;
      this.preview = res;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError' || ctl.signal.aborted) return;
      this.preview = null;
      this.previewError = detectorErrorLines(e).join(' ');
    } finally {
      if (this.#abort === ctl) {
        this.previewing = false;
        this.#abort = null;
      }
    }
  }

  /** The PUT. The page calls this only after the operator confirmed. */
  async save(): Promise<boolean> {
    if (this.revision == null) return false;
    this.saving = true;
    this.saveLines = [];
    this.unknownNames = [];
    this.conflict = false;
    try {
      const res = await this.#deps.putPolicy({
        ...$state.snapshot(this.draft),
        expected_revision: this.revision,
      });
      this.#adopt(res);
      this.unknownNames = res.unknown_names ?? [];
      return true;
    } catch (e) {
      this.saveLines = detectorErrorLines(e);
      this.conflict = detectorErrorDetail(e)?.error === 'revision_conflict';
      return false;
    } finally {
      this.saving = false;
    }
  }

  dispose(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#abort?.abort();
  }
}
