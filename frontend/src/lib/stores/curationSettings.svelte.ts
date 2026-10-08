/**
 * CurationSettingsStore — runes cache for the deployment-wide curation
 * defaults record (`GET,PUT {API_PREFIX}/settings`).
 *
 * Shaped after `strategiesStore` (idempotent `init()`, an `#inflight`
 * promise, a `reset()`, and — critically — ZERO window/document event
 * listeners; it is a data cache, not a UI concern). It differs in two
 * ways, each forced by this endpoint being a WRITE surface rather than
 * read-only capability discovery:
 *
 *  1. `saveDefault()` rethrows. A failed write is the one thing in this
 *     app an operator MUST see; `CLAUDE.md`'s data-integrity rule
 *     ("optimistic UI with error rollback toast on API failure") is
 *     satisfied here by not being optimistic at all.
 *  2. No polling. See the plan's §4.3.
 */

import { getCurationSettings, putCurationDefaults, apiErrorText } from '$lib/api';
import { onProjectChange } from '$lib/projectChange';
import {
  EMPTY_CURATION_SETTINGS,
  axisSpec,
  type CurationSettings,
} from '$lib/curationSettings';

class CurationSettingsStore {
  settings = $state<CurationSettings>(EMPTY_CURATION_SETTINGS);
  loading = $state<boolean>(false);
  loaded = $state<boolean>(false);
  /** Load error. */
  error = $state<string | null>(null);
  /** In-flight axis id during a save, for per-control button state. */
  saving = $state<string | null>(null);

  #inflight: Promise<void> | null = null;
  #gen = 0;

  async init(): Promise<void> {
    if (this.loaded) return;
    if (this.#inflight) return this.#inflight;
    this.loading = true;
    const gen = this.#gen;
    this.#inflight = (async () => {
      try {
        const settings = await getCurationSettings();
        // A load started for the previous project never lands.
        if (gen !== this.#gen) return;
        this.settings = settings;
        this.error = null;
      } catch (e) {
        if (gen !== this.#gen) return;
        if ((e as Error)?.name === 'AbortError') return;
        this.settings = EMPTY_CURATION_SETTINGS;
        this.error = apiErrorText(e) ?? 'failed to load settings';
      } finally {
        if (gen === this.#gen) {
          this.loading = false;
          this.loaded = true;
          this.#inflight = null;
        }
      }
    })();
    return this.#inflight;
  }

  /** Drop the cache and re-read on the next `init()`. Bound to the
   *  page's explicit "Reload" button — the only refresh path, since
   *  there is no poll. */
  reset(): void {
    this.#gen += 1;
    this.loading = false;
    this.settings = EMPTY_CURATION_SETTINGS;
    this.loaded = false;
    this.error = null;
    this.#inflight = null;
  }

  async refresh(): Promise<void> {
    this.reset();
    return this.init();
  }

  /**
   * Pin one axis's shared default.
   *
   * Sends ONLY that axis (`{defaults: {[axis]: id}}`) — a partial merge,
   * per the backend contract. Adopts the server's returned record
   * wholesale rather than mutating `defaults[axis]` locally: the server
   * is authoritative about the merge result, and adopting it is what
   * keeps an axis this build doesn't know about visible in the record.
   *
   * Only an axis this build knows (`SETTINGS_AXES`) is sent. Whether the
   * axis is settable at all is the server's call: `/settings` only offers
   * a control for axes `/methods` marks `settable`, and `PUT /settings`
   * 422s any other.
   *
   * Rethrows on failure after recording `error`, so the caller can toast
   * the server's own `detail`.
   */
  async saveDefault(axis: string, id: string): Promise<void> {
    const spec = axisSpec(axis);
    if (!spec) throw new Error(`unknown settings axis: ${axis}`);
    this.saving = axis;
    try {
      this.settings = await putCurationDefaults({ [axis]: id });
      this.error = null;
    } catch (e) {
      if ((e as Error)?.name !== 'AbortError') {
        this.error = apiErrorText(e) ?? 'failed to save settings';
      }
      throw e;
    } finally {
      this.saving = null;
    }
  }

  /**
   * Clear one axis's pinned shared default, falling back to that axis's
   * own built-in default. Sends `{[axis]: null}` — see
   * `putCurationDefaults`'s docstring for the backend's clear contract.
   *
   * Same known-axis guard as `saveDefault()`.
   */
  async clearDefault(axis: string): Promise<void> {
    const spec = axisSpec(axis);
    if (!spec) throw new Error(`unknown settings axis: ${axis}`);
    this.saving = axis;
    try {
      this.settings = await putCurationDefaults({ [axis]: null });
      this.error = null;
    } catch (e) {
      if ((e as Error)?.name !== 'AbortError') {
        this.error = apiErrorText(e) ?? 'failed to clear setting';
      }
      throw e;
    } finally {
      this.saving = null;
    }
  }
}

export const curationSettingsStore = new CurationSettingsStore();
// Settings are per project: a switch re-reads them.
onProjectChange(() => curationSettingsStore.reset());
