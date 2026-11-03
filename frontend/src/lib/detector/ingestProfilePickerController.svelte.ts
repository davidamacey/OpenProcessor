/**
 * The ingest-profile list-and-activate control on `/settings/ingest-policy`
 * (`GET /ingest_profiles`, `POST /ingest_profiles/{name}/activate`).
 *
 * Thin: the list, the `active` flags and every refusal are the server's.
 * An activation pins the listed revision and sends `expected_active` as
 * last read; a 422 `validation_failed` keeps the served report (the
 * `detector_model_*` and `triton_unreachable` issues render as served), and
 * `force` is sent only when the caller retries after the report said
 * `force_allowed`.
 */
import { apiErrorText } from '$lib/api_errors';
import { activateIngestProfile, listIngestProfiles } from '$lib/api_detector';
import { configErrorDetail } from '$lib/api_profiles';
import type {
  ActivateResponse,
  ConfigActivateRequest,
  ValidationReport,
} from '$lib/types_config';
import type { IngestProfileList } from '$lib/types_detector';

export interface IngestProfilePickerDeps {
  list: (signal?: AbortSignal) => Promise<IngestProfileList>;
  activate: (name: string, body: ConfigActivateRequest) => Promise<ActivateResponse>;
  /** After a successful activation (the page re-reads the active card). */
  onactivated: () => Promise<void> | void;
}

const DEFAULT_DEPS = (onactivated: IngestProfilePickerDeps['onactivated']) => ({
  list: listIngestProfiles,
  activate: activateIngestProfile,
  onactivated,
});

/** A refused activation, as served. */
export interface ActivationRefusal {
  name: string;
  message: string;
  /** The 422 report; null for a refusal without one (e.g. a 409). */
  report: ValidationReport | null;
}

export class IngestProfilePicker {
  list = $state<IngestProfileList | null>(null);
  loadError = $state<string | null>(null);
  /** The profile whose activation is in flight. */
  activating = $state<string | null>(null);
  refusal = $state<ActivationRefusal | null>(null);
  /** The served report of the last successful activation (warnings). */
  lastValidation = $state<ValidationReport | null>(null);

  #deps: IngestProfilePickerDeps;

  constructor(deps: IngestProfilePickerDeps) {
    this.#deps = deps;
  }

  static withDefaults(onactivated: IngestProfilePickerDeps['onactivated']) {
    return new IngestProfilePicker(DEFAULT_DEPS(onactivated));
  }

  async load(signal?: AbortSignal): Promise<void> {
    try {
      this.list = await this.#deps.list(signal);
      this.loadError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.list = null;
      this.loadError = apiErrorText(e);
    }
  }

  async activate(name: string, force = false): Promise<boolean> {
    const list = this.list;
    if (!list || this.activating) return false;
    const target = list.profiles.find((p) => p.name === name);
    if (!target) return false;
    this.activating = name;
    this.refusal = null;
    this.lastValidation = null;
    try {
      const res = await this.#deps.activate(name, {
        revision: target.revision,
        expected_active: list.active,
        force,
      });
      this.lastValidation = res.validation;
      await this.load();
      await this.#deps.onactivated();
      return true;
    } catch (e) {
      const d = configErrorDetail(e);
      this.refusal = {
        name,
        message: d?.message ?? apiErrorText(e),
        report: d?.report ?? null,
      };
      if (d?.error === 'active_conflict') await this.load();
      return false;
    } finally {
      this.activating = null;
    }
  }
}
