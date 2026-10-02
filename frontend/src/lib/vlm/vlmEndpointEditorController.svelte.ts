/**
 * One VLM endpoint's editor (any_domain_plan.md §7.8.5;
 * docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md §3.3): the VLM binding of
 * the shared `ConfigEditor` (draft, served live validation, Save with
 * `expected_revision`, revisions and restore) over the GLOBAL registry
 * routes, plus
 *
 * - the registry list (for `secret_refs`) and the catalog (for
 *   `vlm_catalog` choices) the form's pickers render from;
 * - `draftFacts` / "Test connection" (`validate?probe=true`, `name: null`);
 * - "Probe saved" (`POST /vlm/endpoints/{name}/probe`, then the doc is
 *   re-read for `last_probe` without touching the draft);
 * - this project's activation (`VlmActive`).
 *
 * The validate adapter returns the served `validation` to the editor and
 * keeps the response's facts. The wake-up follows `vlm.changed`
 * (`registry`) and the project's `config.changed axis=vlm`.
 */
import { configErrorText } from '$lib/api';
import {
  getVlmCatalog,
  getVlmEndpoint,
  getVlmEndpointRevision,
  getVlmEndpointRevisions,
  getVlmEndpointSchema,
  listVlmEndpoints,
  probeVlmEndpoint,
  updateVlmEndpoint,
  validateVlmEndpoint,
} from '$lib/api_vlm';
import { ConfigEditor } from '$lib/config/configEditor.svelte';
import type {
  VlmCatalogResponse,
  VlmEndpointBody,
  VlmEndpointDoc,
  VlmEndpointList,
  VlmEndpointSchema,
  VlmProbeResult,
} from '$lib/types_vlm';
import { VlmActive, type VlmActiveDeps } from './vlmActive.svelte';
import { VlmDraftChecks } from './vlmDraft.svelte';
import {
  isVlmActiveEvent,
  isVlmRegistryEvent,
  subscribeVlmEvents,
  type VlmEvent,
} from './vlmEvents';
import { normalizeFieldValue } from './vlmFields';

export interface VlmEditorDeps extends VlmActiveDeps {
  getVlmEndpoint: typeof getVlmEndpoint;
  getVlmEndpointSchema: typeof getVlmEndpointSchema;
  getVlmEndpointRevisions: typeof getVlmEndpointRevisions;
  getVlmEndpointRevision: typeof getVlmEndpointRevision;
  updateVlmEndpoint: typeof updateVlmEndpoint;
  validateVlmEndpoint: typeof validateVlmEndpoint;
  probeVlmEndpoint: typeof probeVlmEndpoint;
  listVlmEndpoints: typeof listVlmEndpoints;
  getVlmCatalog: typeof getVlmCatalog;
  subscribe: (onEvent: (e: VlmEvent) => void) => { close(): void };
}

export class VlmEndpointEditor extends ConfigEditor<
  VlmEndpointBody,
  VlmEndpointDoc,
  VlmEndpointSchema,
  VlmActive,
  VlmEvent
> {
  /** The registry list: `secret_refs` and the served labels. */
  list = $state<VlmEndpointList | null>(null);
  catalog = $state<VlmCatalogResponse | null>(null);
  extrasError = $state<string | null>(null);

  readonly checks: VlmDraftChecks;
  /** The last "Probe saved" result. */
  probeResult = $state<VlmProbeResult | null>(null);
  probing = $state(false);
  probeError = $state<string | null>(null);

  #deps: Partial<VlmEditorDeps>;

  constructor(name: string, deps: Partial<VlmEditorDeps> = {}) {
    const checks = new VlmDraftChecks(deps.validateVlmEndpoint ?? validateVlmEndpoint);
    super(
      name,
      {
        getSchema: () => (deps.getVlmEndpointSchema ?? getVlmEndpointSchema)(),
        getDoc: (n) => (deps.getVlmEndpoint ?? getVlmEndpoint)(n),
        getRevisions: (n) => (deps.getVlmEndpointRevisions ?? getVlmEndpointRevisions)(n),
        getRevision: (n, r) =>
          (deps.getVlmEndpointRevision ?? getVlmEndpointRevision)(n, r),
        update: (n, body) => (deps.updateVlmEndpoint ?? updateVlmEndpoint)(n, body),
        // `name: null`: the endpoint's own name would read as taken.
        validate: (body, signal) => checks.validate(body.name, body.body, signal),
        subscribe: (onEvent) =>
          (deps.subscribe ?? subscribeVlmEvents)((e) => {
            // A registry write may carry no endpoint name (A-4): treat it
            // as possibly this one, and re-read.
            if (isVlmRegistryEvent(e) && (e as { name?: unknown }).name == null) {
              onEvent({ ...e, name } as VlmEvent);
            } else {
              onEvent(e);
            }
          }),
        isEvent: (e) => isVlmRegistryEvent(e) || isVlmActiveEvent(e),
      },
      new VlmActive(deps),
    );
    this.checks = checks;
    this.#deps = deps;
  }

  protected override async loadExtras(): Promise<void> {
    const d = this.#deps;
    const [list, catalog] = await Promise.allSettled([
      (d.listVlmEndpoints ?? listVlmEndpoints)(),
      (d.getVlmCatalog ?? getVlmCatalog)(),
    ]);
    // A failure here is shown, never fatal: the pickers then fall back to
    // a text input.
    this.list = list.status === 'fulfilled' ? list.value : null;
    this.catalog = catalog.status === 'fulfilled' ? catalog.value : null;
    const failed = [list, catalog].find((r) => r.status === 'rejected');
    this.extrasError = failed
      ? configErrorText((failed as PromiseRejectedResult).reason)
      : null;
  }

  override setField(field: string, value: unknown): void {
    super.setField(
      field,
      normalizeFieldValue(
        this.schema,
        field,
        value,
      ) as VlmEndpointBody[keyof VlmEndpointBody],
    );
    this.checks.clearProbe();
  }

  /** "Test connection": the draft with `probe=true`; the served report
   *  replaces the one on screen. */
  async testConnection(): Promise<void> {
    const report = await this.checks.test(null, this.draftBody);
    if (report) this.report = report;
  }

  /** "Probe saved": a live probe of the saved endpoint; the doc is then
   *  re-read for `last_probe` (the draft is untouched). */
  async probeSaved(): Promise<void> {
    if (this.probing) return;
    this.probing = true;
    this.probeError = null;
    try {
      this.probeResult = await (this.#deps.probeVlmEndpoint ?? probeVlmEndpoint)(
        this.name,
      );
      this.doc = await (this.#deps.getVlmEndpoint ?? getVlmEndpoint)(this.name);
    } catch (e) {
      this.probeError = configErrorText(e);
    } finally {
      this.probing = false;
    }
  }
}

export function createVlmEndpointEditor(
  name: string,
  deps: Partial<VlmEditorDeps> = {},
): VlmEndpointEditor {
  return new VlmEndpointEditor(name, deps);
}
