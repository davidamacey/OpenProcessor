/**
 * `packsAvailability` — the "not yet deployed" gate for OpenProcessor W3
 * (prompt-pack CRUD).
 *
 * This is NOT backward compatibility. The spec gives W3 no capability
 * signal (no `/health` flag, and `/methods`' `prompt_pack` axis predates
 * W3), so until W3 is deployed the one way to know whether its routes
 * exist is to ask: a 404/501 on `GET {prefix}/prompt_packs` means "not
 * deployed" and every pack surface is absent. Any other failure keeps
 * `available` unknown and records the error, so a transient outage never
 * hides a route that exists (docs/design/w3-pack-editor-ui-plan-2026-09-27.md
 * §0.3, question W3-Q1).
 *
 * To remove the gate once W3 ships everywhere: drop the 404 branch and
 * `available`.
 *
 * Probed lazily (the pack surfaces and the `/settings` card call
 * `init()`), at most once per project; reset on a project switch.
 */
import { listPromptPacks, type ApiError } from '$lib/api';
import { onProjectChange } from '$lib/projectChange';

class PacksAvailabilityStore {
  /** `null` until the probe answers. */
  available = $state<boolean | null>(null);
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
        await listPromptPacks();
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
        this.error = err?.detail ?? err?.message ?? String(e);
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

export const packsAvailability = new PacksAvailabilityStore();

onProjectChange(() => packsAvailability.reset());
