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
import { ConfigAvailability } from '$lib/config/configAvailability.svelte';
import { onProjectChange } from '$lib/projectChange';
import type { DatasetFormatsResponse } from '$lib/types_import';

class DatasetsAvailabilityStore extends ConfigAvailability<DatasetFormatsResponse> {
  formats = $state<DatasetFormatsResponse | null>(null);

  constructor() {
    super(getDatasetFormats);
  }

  protected override accept(value: DatasetFormatsResponse): void {
    this.formats = value;
  }

  protected override clear(): void {
    this.formats = null;
  }

  /** Served status label, else the raw id. */
  statusLabel(status: string): string {
    return this.formats?.status_labels?.[status] ?? status;
  }
}

export const datasetsAvailability = new DatasetsAvailabilityStore();

onProjectChange(() => datasetsAvailability.reset());
