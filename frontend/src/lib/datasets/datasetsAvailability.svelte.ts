/**
 * `datasetsAvailability` — the "not yet deployed" gate for dataset import
 * and Reprocess, plus the served vocabulary every import
 * control renders from (`GET {prefix}/datasets/formats`).
 *
 * This is NOT backward compatibility. The `/health api_features` signal
 * was dropped, so until dataset import is deployed the
 * one way to know whether its routes exist is to ask: a 404/501 on
 * `/datasets/formats` means "not deployed" and every import surface is
 * absent. Any other failure keeps `available` unknown and records the
 * error, so a transient outage never hides a route that exists.
 *
 * To remove the gate once dataset import ships everywhere: drop the 404 branch and
 * `available`, and keep this as a plain vocabulary loader.
 *
 * Probed lazily (the surfaces that need it call `init()`), at most once
 * per project; reset on a project switch.
 */
import { getDatasetFormats } from '$lib/api_datasetImport';
import { apiErrorText, type ApiError } from '$lib/api_errors';
import { onProjectChange } from '$lib/projectChange';
import type { DatasetFormatsResponse } from '$lib/types_import';

class DatasetsAvailabilityStore {
  /** `null` until the probe answers. */
  available = $state<boolean | null>(null);
  formats = $state<DatasetFormatsResponse | null>(null);
  error = $state<string | null>(null);
  #inflight: Promise<void> | null = null;
  #loaded = false;
  #generation = 0;

  init(): Promise<void> {
    if (this.#loaded) return Promise.resolve();
    if (this.#inflight) return this.#inflight;
    const gen = this.#generation;
    this.#inflight = (async () => {
      try {
        const formats = await getDatasetFormats();
        if (gen !== this.#generation) return;
        this.formats = formats;
        this.available = true;
        this.error = null;
        this.#loaded = true;
      } catch (e) {
        if (gen !== this.#generation) return;
        if ((e as Error)?.name === 'AbortError') return;
        const err = e as Partial<ApiError>;
        if (err?.status === 404 || err?.status === 501) {
          this.available = false;
          this.#loaded = true;
          return;
        }
        this.error = apiErrorText(e);
      } finally {
        if (gen === this.#generation) this.#inflight = null;
      }
    })();
    return this.#inflight;
  }

  /** Re-probe after a failure other than 404 (the retry button). */
  retry(): Promise<void> {
    this.error = null;
    this.#loaded = false;
    return this.init();
  }

  /** Served status label, else the raw id. */
  statusLabel(status: string): string {
    return this.formats?.status_labels?.[status] ?? status;
  }

  reset(): void {
    this.#generation += 1;
    this.#inflight = null;
    this.#loaded = false;
    this.available = null;
    this.formats = null;
    this.error = null;
  }
}

export const datasetsAvailability = new DatasetsAvailabilityStore();

onProjectChange(() => datasetsAvailability.reset());
