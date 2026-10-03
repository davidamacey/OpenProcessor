/**
 * "Create classes from the detector" (`POST /classes/seed_from_detector`).
 *
 * A dry run first (`preview()`), then, behind the page's confirm, the real
 * call (`create()`) with the same chosen names. Nothing here decides which
 * classes a detector label becomes: the lists are the served ones.
 */
import { getIngestConfig } from '$lib/api';
import { detectorErrorLines, seedFromDetector } from '$lib/api_detector';
import type {
  IngestDetectorInfo,
  SeedFromDetectorRequest,
  SeedFromDetectorResponse,
} from '$lib/types_detector';
import { classesStore } from '$stores/classes.svelte';

export interface SeedDeps {
  getConfig: (signal?: AbortSignal) => Promise<{ detector?: IngestDetectorInfo | null }>;
  seed: (
    req: SeedFromDetectorRequest & { dry_run: boolean },
  ) => Promise<SeedFromDetectorResponse>;
  /** Called after a real create succeeded (reloads the class registry). */
  onSeeded: () => Promise<void>;
}

const DEFAULT_DEPS: SeedDeps = {
  getConfig: getIngestConfig,
  seed: seedFromDetector,
  onSeeded: () => classesStore.clearAndRefetch(),
};

export class SeedFromDetector {
  detector = $state<IngestDetectorInfo | null>(null);
  loaded = $state(false);
  /** Detector label names the operator chose; empty = all. */
  chosen = $state<string[]>([]);
  /** The served dry run, until the choice changes or a create ran. */
  result = $state<SeedFromDetectorResponse | null>(null);
  /** The served result of the last real create. */
  created = $state<SeedFromDetectorResponse | null>(null);
  busy = $state(false);
  errorLines = $state<string[]>([]);

  #deps: SeedDeps;
  /** The names the shown preview was computed for. */
  #previewedNames: string[] | null = null;

  constructor(deps: SeedDeps = DEFAULT_DEPS) {
    this.#deps = deps;
  }

  get available(): boolean {
    return this.detector !== null;
  }

  get labelNames(): string[] {
    return this.detector?.labels.map((l) => l.name) ?? [];
  }

  async load(signal?: AbortSignal): Promise<void> {
    try {
      this.detector = (await this.#deps.getConfig(signal)).detector ?? null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.errorLines = detectorErrorLines(e);
    } finally {
      this.loaded = true;
    }
  }

  setChosen(names: string[]): void {
    this.chosen = names;
    this.result = null;
    this.#previewedNames = null;
  }

  #request(
    dry_run: boolean,
    names: string[],
  ): SeedFromDetectorRequest & { dry_run: boolean } {
    return names.length > 0 ? { dry_run, names } : { dry_run };
  }

  async preview(): Promise<void> {
    this.busy = true;
    this.errorLines = [];
    this.created = null;
    const names = [...this.chosen];
    try {
      this.result = await this.#deps.seed(this.#request(true, names));
      this.#previewedNames = names;
    } catch (e) {
      this.result = null;
      this.#previewedNames = null;
      this.errorLines = detectorErrorLines(e);
    } finally {
      this.busy = false;
    }
  }

  /** The real call. The page calls this only after the operator confirmed
   *  the preview; it never runs without one. */
  async create(): Promise<boolean> {
    if (this.result === null || this.#previewedNames === null) return false;
    this.busy = true;
    this.errorLines = [];
    try {
      this.created = await this.#deps.seed(this.#request(false, this.#previewedNames));
      this.result = null;
      this.#previewedNames = null;
      await this.#deps.onSeeded();
      return true;
    } catch (e) {
      this.errorLines = detectorErrorLines(e);
      return false;
    } finally {
      this.busy = false;
    }
  }
}
