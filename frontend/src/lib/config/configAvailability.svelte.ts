/**
 * The "not yet deployed" gate for a config-store editor (OpenProcessor W3
 * prompt packs, W4 region profiles).
 *
 * This is NOT backward compatibility. Neither wave serves a capability
 * signal (docs/design/w3-pack-editor-ui-plan-2026-09-27.md §0.3, W3-Q1;
 * docs/design/w4-profile-editor-ui-plan-2026-09-27.md §0.3, W4-Q1), so
 * until the wave is deployed the one way to know whether its routes exist
 * is to ask: a 404/501 on the probe means "not deployed" and every surface
 * of that editor is absent. Any other failure keeps `available` unknown
 * and records the error, so a transient outage never hides a route that
 * exists.
 *
 * To remove a gate once its wave ships everywhere: drop the 404 branch
 * and `available`.
 *
 * Probed lazily (the editor surfaces call `init()`), at most once per
 * project; each instance's module registers its reset on project change.
 */
import { apiErrorText, type ApiError } from '$lib/api';

export class ConfigAvailability {
  /** `null` until the probe answers. */
  available = $state<boolean | null>(null);
  error = $state<string | null>(null);
  #probe: () => Promise<unknown>;
  #inflight: Promise<void> | null = null;
  #loaded = false;
  #generation = 0;

  constructor(probe: () => Promise<unknown>) {
    this.#probe = probe;
  }

  init(): Promise<void> {
    if (this.#loaded) return Promise.resolve();
    if (this.#inflight) return this.#inflight;
    const gen = this.#generation;
    this.#inflight = (async () => {
      try {
        await this.#probe();
        if (gen !== this.#generation) return;
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

  /** Re-probe after a failure other than 404 (the Retry button). */
  retry(): Promise<void> {
    this.error = null;
    this.#loaded = false;
    return this.init();
  }

  reset(): void {
    this.#generation += 1;
    this.#inflight = null;
    this.#loaded = false;
    this.available = null;
    this.error = null;
  }
}
