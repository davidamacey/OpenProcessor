/**
 * The "new endpoint" page's state (any_domain_plan.md §7.8.5;
 * docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md §3.3). The shared
 * `ConfigEditor` edits an existing doc, so create is its own small
 * controller: a name, a description and a body seeded from each served
 * schema row's `default`; live validation (400 ms after an edit) posts
 * `{name: <typed name>, body}` so the served `name_conflict` / `vlm_name_*`
 * issues surface before Create; "Test connection" probes the draft;
 * Create posts `{name, description, body}`. A 409 `name_conflict` / 422
 * `validation_failed` shows the served message and report.
 */
import { configErrorDetail, configErrorText } from '$lib/api';
import {
  createVlmEndpoint,
  getVlmCatalog,
  getVlmEndpointSchema,
  listVlmEndpoints,
  validateVlmEndpoint,
} from '$lib/api_vlm';
import { VALIDATE_DEBOUNCE_MS } from '$lib/config/configEditor.svelte';
import type { ValidationReport } from '$lib/types_config';
import type {
  VlmCatalogResponse,
  VlmEndpointBody,
  VlmEndpointDoc,
  VlmEndpointList,
  VlmEndpointSchema,
} from '$lib/types_vlm';
import { VlmDraftChecks } from './vlmDraft.svelte';
import { bodyFromDefaults, normalizeFieldValue } from './vlmFields';

export interface VlmCreateDeps {
  getVlmEndpointSchema: typeof getVlmEndpointSchema;
  listVlmEndpoints: typeof listVlmEndpoints;
  getVlmCatalog: typeof getVlmCatalog;
  validateVlmEndpoint: typeof validateVlmEndpoint;
  createVlmEndpoint: typeof createVlmEndpoint;
}

export class VlmEndpointCreator {
  schema = $state<VlmEndpointSchema | null>(null);
  list = $state<VlmEndpointList | null>(null);
  catalog = $state<VlmCatalogResponse | null>(null);
  loadError = $state<string | null>(null);

  name = $state('');
  description = $state('');
  draftBody = $state<VlmEndpointBody>({} as VlmEndpointBody);

  report = $state<ValidationReport | null>(null);
  validating = $state(false);
  validateError = $state<string | null>(null);

  creating = $state(false);
  createError = $state<string | null>(null);
  createReport = $state<ValidationReport | null>(null);

  readonly checks: VlmDraftChecks;

  #deps: Partial<VlmCreateDeps>;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #abort: AbortController | null = null;

  constructor(deps: Partial<VlmCreateDeps> = {}) {
    this.#deps = deps;
    this.checks = new VlmDraftChecks(deps.validateVlmEndpoint ?? validateVlmEndpoint);
  }

  async load(): Promise<void> {
    const d = this.#deps;
    try {
      const [schema, list, catalog] = await Promise.allSettled([
        (d.getVlmEndpointSchema ?? getVlmEndpointSchema)(),
        (d.listVlmEndpoints ?? listVlmEndpoints)(),
        (d.getVlmCatalog ?? getVlmCatalog)(),
      ]);
      if (schema.status === 'rejected') throw schema.reason;
      this.schema = schema.value;
      this.draftBody = bodyFromDefaults(schema.value);
      // The pickers fall back to a text input when a list is missing.
      this.list = list.status === 'fulfilled' ? list.value : null;
      this.catalog = catalog.status === 'fulfilled' ? catalog.value : null;
      this.loadError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = configErrorText(e);
    }
  }

  stop(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
    this.#abort?.abort();
  }

  /** The typed name, or null while empty (nothing to check yet). */
  get typedName(): string | null {
    const n = this.name.trim();
    return n === '' ? null : n;
  }

  get canCreate(): boolean {
    return this.typedName != null && !this.creating && this.schema != null;
  }

  setName(value: string): void {
    this.name = value;
    this.createError = null;
    this.scheduleValidate();
  }

  setDescription(value: string): void {
    this.description = value;
  }

  setField(field: string, value: unknown): void {
    this.draftBody = {
      ...this.draftBody,
      [field]: normalizeFieldValue(this.schema, field, value),
    };
    this.checks.clearProbe();
    this.scheduleValidate();
  }

  scheduleValidate(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = setTimeout(() => {
      this.#timer = null;
      void this.validateNow();
    }, VALIDATE_DEBOUNCE_MS);
  }

  async validateNow(): Promise<void> {
    this.#abort?.abort();
    const ctl = new AbortController();
    this.#abort = ctl;
    this.validating = true;
    try {
      const report = await this.checks.validate(
        this.typedName,
        this.draftBody,
        ctl.signal,
      );
      if (ctl.signal.aborted) return;
      this.report = report;
      this.validateError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.validateError = configErrorText(e);
    } finally {
      if (this.#abort === ctl) {
        this.validating = false;
        this.#abort = null;
      }
    }
  }

  async testConnection(): Promise<void> {
    const report = await this.checks.test(this.typedName, this.draftBody);
    if (report) this.report = report;
  }

  /** Creates the endpoint; the new doc, or null (the served refusal is in
   *  `createError` / `createReport`). */
  async create(): Promise<VlmEndpointDoc | null> {
    const name = this.typedName;
    if (!name || this.creating) return null;
    this.creating = true;
    this.createError = null;
    this.createReport = null;
    try {
      return await (this.#deps.createVlmEndpoint ?? createVlmEndpoint)({
        name,
        description: this.description.trim(),
        body: this.draftBody,
      });
    } catch (e) {
      this.createError = configErrorText(e);
      this.createReport = configErrorDetail(e)?.report ?? null;
      return null;
    } finally {
      this.creating = false;
    }
  }
}

export function createVlmEndpointCreator(
  deps: Partial<VlmCreateDeps> = {},
): VlmEndpointCreator {
  return new VlmEndpointCreator(deps);
}
