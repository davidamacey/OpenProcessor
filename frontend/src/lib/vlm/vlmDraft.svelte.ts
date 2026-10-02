/**
 * What the server says about an UNSAVED endpoint draft, shared by the
 * editor and the create page: the facts `POST /vlm/endpoints/validate`
 * returns beside its report (`locality`, `sends_images_externally`) and
 * the result of "Test connection" (`validate?probe=true`, whose served
 * `probe` is shown as is; a 429 `probe_busy` is the served message).
 * Nothing here judges a result.
 */
import { configErrorText } from '$lib/api';
import { validateVlmEndpoint } from '$lib/api_vlm';
import type {
  VlmEndpointBody,
  VlmLocality,
  VlmProbeResult,
  VlmValidateResponse,
} from '$lib/types_vlm';
import type { ValidationReport } from '$lib/types_config';

export interface DraftFacts {
  locality: VlmLocality | null;
  sends_images_externally: boolean;
}

export class VlmDraftChecks {
  facts = $state<DraftFacts | null>(null);
  probe = $state<VlmProbeResult | null>(null);
  testing = $state(false);
  testError = $state<string | null>(null);

  #validate: typeof validateVlmEndpoint;

  constructor(validate: typeof validateVlmEndpoint = validateVlmEndpoint) {
    this.#validate = validate;
  }

  /** Posts the draft (live validation, no probe) and keeps its facts. */
  async validate(
    name: string | null,
    body: VlmEndpointBody,
    signal?: AbortSignal,
  ): Promise<ValidationReport> {
    const res = await this.#validate({ name, body }, false, signal);
    this.#adopt(res);
    return res.validation;
  }

  /** "Test connection": the draft with `probe=true`. The served report
   *  (null on a refusal) so the caller can place its issues. */
  async test(
    name: string | null,
    body: VlmEndpointBody,
  ): Promise<ValidationReport | null> {
    this.testing = true;
    this.testError = null;
    try {
      const res = await this.#validate({ name, body }, true);
      this.#adopt(res);
      this.probe = res.probe ?? null;
      return res.validation;
    } catch (e) {
      this.testError = configErrorText(e);
      return null;
    } finally {
      this.testing = false;
    }
  }

  /** The draft changed: a shown probe was for a different one. */
  clearProbe(): void {
    this.probe = null;
    this.testError = null;
  }

  #adopt(res: VlmValidateResponse): void {
    this.facts = {
      locality: res.locality,
      sends_images_externally: res.sends_images_externally,
    };
  }
}
