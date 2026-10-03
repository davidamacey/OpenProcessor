/**
 * The served `GET /detections/summary` for the dashboard panel. Reads and
 * holds it as served; the "Embed N" request is the summary's own
 * `suggested_reprocess`, never composed here.
 */
import { detectorErrorLines, getDetectionsSummary } from '$lib/api_detector';
import type { DetectionsSummary } from '$lib/types_detector';

export class DetectionsSummaryState {
  summary = $state<DetectionsSummary | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  #get: (signal?: AbortSignal) => Promise<DetectionsSummary>;

  constructor(
    get: (signal?: AbortSignal) => Promise<DetectionsSummary> = getDetectionsSummary_,
  ) {
    this.#get = get;
  }

  async load(signal?: AbortSignal): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      this.summary = await this.#get(signal);
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.summary = null;
      this.error = detectorErrorLines(e).join(' ');
    } finally {
      this.loading = false;
    }
  }
}

function getDetectionsSummary_(signal?: AbortSignal): Promise<DetectionsSummary> {
  return getDetectionsSummary({}, signal);
}
