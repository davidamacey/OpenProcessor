/**
 * CurationSettingsStore — runes cache for the deployment-wide curation
 * defaults record (`GET,PUT {API_PREFIX}/settings`).
 *
 * Shaped after `strategiesStore` (idempotent `init()`, an `#inflight`
 * promise, a `reset()`, and — critically — ZERO window/document event
 * listeners; it is a data cache, not a UI concern). It differs in three
 * ways, each forced by this endpoint being a WRITE surface rather than
 * read-only capability discovery:
 *
 *  1. It has a `supported` tri-state. `getMethods()` can collapse every
 *     failure into a fallback because a missing capability list has a
 *     correct default; a missing settings record does not — "this
 *     backend has no shared defaults" and "the request failed" need
 *     different words on screen and different affordances.
 *  2. `saveDefault()` rethrows. A failed write is the one thing in this
 *     app an operator MUST see; `CLAUDE.md`'s data-integrity rule
 *     ("optimistic UI with error rollback toast on API failure") is
 *     satisfied here by not being optimistic at all.
 *  3. No polling. See the plan's §4.3.
 */

import { getCurationSettings, putCurationDefaults, ApiError } from '$lib/api';
import {
  EMPTY_CURATION_SETTINGS,
  axisSpec,
  type CurationSettings,
} from '$lib/curationSettings';

class CurationSettingsStore {
  settings = $state<CurationSettings>(EMPTY_CURATION_SETTINGS);
  loading = $state<boolean>(false);
  loaded = $state<boolean>(false);
  /** `null` = not determined yet. `false` = the backend 404'd this route
   *  (predates the feature). `true` = a real record was read. */
  supported = $state<boolean | null>(null);
  /** Load error, distinct from `supported === false`. */
  error = $state<string | null>(null);
  /** In-flight axis id during a save, for per-control button state. */
  saving = $state<string | null>(null);

  #inflight: Promise<void> | null = null;

  async init(): Promise<void> {
    if (this.loaded) return;
    if (this.#inflight) return this.#inflight;
    this.loading = true;
    this.#inflight = (async () => {
      try {
        this.settings = await getCurationSettings();
        this.supported = true;
        this.error = null;
      } catch (e) {
        if ((e as Error)?.name === 'AbortError') return;
        this.settings = EMPTY_CURATION_SETTINGS;
        if (e instanceof ApiError && e.status === 404) {
          this.supported = false;
          this.error = null;
        } else {
          this.supported = null;
          this.error = (e as Error)?.message ?? 'failed to load settings';
        }
      } finally {
        this.loading = false;
        this.loaded = true;
        this.#inflight = null;
      }
    })();
    return this.#inflight;
  }

  /** Drop the cache and re-read on the next `init()`. Bound to the
   *  page's explicit "Reload" button — the only refresh path, since
   *  there is no poll. */
  reset(): void {
    this.settings = EMPTY_CURATION_SETTINGS;
    this.loaded = false;
    this.supported = null;
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
   * Refuses an advisory axis BEFORE any network call. The backend would
   * happily accept `detection_profile`/`prompt_pack` (they are in
   * `SETTABLE_DEFAULT_AXES`) and store a value nothing honors — a 200
   * that means nothing changed is strictly worse than a 422. This guard
   * is what makes the honesty boundary structural rather than a
   * rendering choice; `SETTINGS_AXES` is its single source of truth.
   *
   * Rethrows on failure after recording `error`, so the caller can toast
   * the server's own `detail`.
   */
  async saveDefault(axis: string, id: string): Promise<void> {
    const spec = axisSpec(axis);
    if (!spec) throw new Error(`unknown settings axis: ${axis}`);
    if (spec.kind !== 'settable') {
      throw new Error(
        `axis '${axis}' is advertised but not honored by any backend code ` +
          `path yet — Cropwright will not write a default that does nothing`,
      );
    }
    this.saving = axis;
    try {
      this.settings = await putCurationDefaults({ [axis]: id });
      this.error = null;
      this.supported = true;
    } catch (e) {
      if ((e as Error)?.name !== 'AbortError') {
        this.error = (e as Error)?.message ?? 'failed to save settings';
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
   * Same advisory-axis guard as `saveDefault()`: there is nothing to
   * clear on an axis this build never let you pin in the first place.
   */
  async clearDefault(axis: string): Promise<void> {
    const spec = axisSpec(axis);
    if (!spec) throw new Error(`unknown settings axis: ${axis}`);
    if (spec.kind !== 'settable') {
      throw new Error(`axis '${axis}' is advisory — there is no pinned default to clear`);
    }
    this.saving = axis;
    try {
      this.settings = await putCurationDefaults({ [axis]: null });
      this.error = null;
      this.supported = true;
    } catch (e) {
      if ((e as Error)?.name !== 'AbortError') {
        this.error = (e as Error)?.message ?? 'failed to clear setting';
      }
      throw e;
    } finally {
      this.saving = null;
    }
  }
}

export const curationSettingsStore = new CurationSettingsStore();
