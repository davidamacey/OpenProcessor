/**
 * The `/settings/models` page's state (any_domain_plan.md §7.8;
 * docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md §3.3): the VLM binding of
 * the shared `ConfigList` (the served registry, this project's active
 * endpoint with Rollback and Turn off, Clone and Delete) plus
 *
 * - Activate here goes through `active` (`VlmActive`): `acknowledge_external`
 *   only when the operator checked it, `force` only after a served
 *   `force_allowed`; the page re-reads the registry (`loadRegistry`) after;
 * - Probe per endpoint (`POST /vlm/endpoints/{name}/probe`);
 * - the local-model catalog: Switch (`POST /vlm/local/select`, with
 *   `force` only after a served `vlm_catalog_does_not_fit`), Cancel the
 *   request, and a re-read of `GET /vlm/local` every served
 *   `poll_after_s` until it is null. Nothing here says a switch happened;
 *   `serving` flips when the server says so;
 * - the "all model choices" table off `GET /config/vocabulary`.
 *
 * Wake-ups, not state: `vlm.changed` `registry` re-reads the registry,
 * `local_vlm` re-reads the local status, and the project's
 * `config.changed axis=vlm` re-reads the active ref.
 */
import { configErrorDetail, apiErrorText, getConfigVocabulary } from '$lib/api';
import {
  clearLocalVlmSelection,
  cloneVlmEndpoint,
  deleteVlmEndpoint,
  getLocalVlm,
  getVlmCatalog,
  listVlmEndpoints,
  probeVlmEndpoint,
  selectLocalVlm,
} from '$lib/api_vlm';
import { ConfigList } from '$lib/config/configList.svelte';
import type { ModelChoice } from '$lib/types_profiles';
import type {
  VlmCatalogResponse,
  VlmEndpointDoc,
  VlmEndpointList,
  VlmLocalStatus,
  VlmProbeResult,
} from '$lib/types_vlm';
import { VlmActive, type VlmActiveDeps } from './vlmActive.svelte';
import {
  isVlmActiveEvent,
  isVlmLocalEvent,
  isVlmRegistryEvent,
  subscribeVlmEvents,
  type VlmEvent,
} from './vlmEvents';

export interface VlmModelsDeps extends VlmActiveDeps {
  listVlmEndpoints: typeof listVlmEndpoints;
  cloneVlmEndpoint: typeof cloneVlmEndpoint;
  deleteVlmEndpoint: typeof deleteVlmEndpoint;
  probeVlmEndpoint: typeof probeVlmEndpoint;
  getVlmCatalog: typeof getVlmCatalog;
  getLocalVlm: typeof getLocalVlm;
  selectLocalVlm: typeof selectLocalVlm;
  clearLocalVlmSelection: typeof clearLocalVlmSelection;
  getConfigVocabulary: typeof getConfigVocabulary;
  subscribe: (onEvent: (e: VlmEvent) => void) => { close(): void };
}

export class VlmModels extends ConfigList<
  VlmEndpointList,
  VlmEndpointDoc,
  VlmActive,
  VlmEvent
> {
  catalog = $state<VlmCatalogResponse | null>(null);
  catalogError = $state<string | null>(null);

  /** The served projects naming an endpoint a Delete was refused for
   *  (409 `in_use`). */
  deleteProjects = $state<string[]>([]);

  probes = $state<Record<string, VlmProbeResult>>({});
  probeErrors = $state<Record<string, string>>({});
  probing = $state<string | null>(null);

  /** The last refused local select/clear, as served. */
  localError = $state<string | null>(null);
  /** The code of that refusal (`vlm_catalog_does_not_fit` offers "Select
   *  anyway"). */
  localErrorCode = $state<string | null>(null);
  localBusy = $state(false);

  modelChoices = $state<ModelChoice[] | null>(null);
  modelChoiceLabels = $state<Record<string, string>>({});
  modelChoicesError = $state<string | null>(null);

  #deps: Partial<VlmModelsDeps>;
  #events: { close(): void } | null = null;
  #pollTimer: ReturnType<typeof setTimeout> | null = null;
  #stopped = false;

  constructor(deps: Partial<VlmModelsDeps> = {}) {
    // The delete adapter records the served `projects[]` of an `in_use`
    // refusal, then rethrows for `ConfigList.remove` to show its message.
    let setProjects: (projects: string[]) => void = () => {};
    super(
      {
        list: () => (deps.listVlmEndpoints ?? listVlmEndpoints)(),
        clone: (name, body) =>
          (deps.cloneVlmEndpoint ?? cloneVlmEndpoint)(name, {
            new_name: body.new_name,
            revision: body.revision,
            description: body.description,
          }),
        remove: async (name, rev) => {
          setProjects([]);
          try {
            await (deps.deleteVlmEndpoint ?? deleteVlmEndpoint)(name, rev);
          } catch (e) {
            setProjects(configErrorDetail(e)?.projects ?? []);
            throw e;
          }
        },
        // Events are routed by `start()` below, not through the shared
        // list's single re-read.
        subscribe: () => ({ close() {} }),
        isEvent: () => false,
      },
      new VlmActive(deps),
    );
    setProjects = (projects) => {
      this.deleteProjects = projects;
    };
    this.#deps = deps;
  }

  get local(): VlmLocalStatus | null {
    return this.catalog?.local ?? null;
  }

  override start(): void {
    this.#stopped = false;
    void this.load();
    void this.loadModelChoices();
    this.#events = (this.#deps.subscribe ?? subscribeVlmEvents)((e) => {
      if (isVlmRegistryEvent(e)) void this.loadRegistry();
      else if (isVlmLocalEvent(e)) void this.loadCatalog();
      else if (isVlmActiveEvent(e)) void this.active.load();
    });
  }

  override stop(): void {
    this.#stopped = true;
    this.#events?.close();
    this.#events = null;
    this.#clearPoll();
    super.stop();
  }

  override async load(): Promise<void> {
    await Promise.all([super.load(), this.loadCatalog()]);
  }

  /** The registry list alone (a `registry` wake-up). */
  async loadRegistry(): Promise<void> {
    try {
      this.list = await this.backend.list();
      this.loadError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = apiErrorText(e);
    }
  }

  /** The catalog and the local status; arms the poll from the served
   *  `poll_after_s`. */
  async loadCatalog(): Promise<void> {
    try {
      this.catalog = await (this.#deps.getVlmCatalog ?? getVlmCatalog)();
      this.catalogError = null;
      this.#armPoll();
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.catalogError = apiErrorText(e);
    }
  }

  /** `GET /vlm/local` alone, for the poll. */
  async pollLocal(): Promise<void> {
    try {
      const local = await (this.#deps.getLocalVlm ?? getLocalVlm)();
      if (this.catalog) this.catalog = { ...this.catalog, local };
      this.#armPoll();
      // `serving` / `desired` flags on the rows moved with it.
      if (this.catalog && !local.restart_required) await this.loadCatalog();
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.catalogError = apiErrorText(e);
      // Keep following the last served `poll_after_s`.
      this.#armPoll();
    }
  }

  #clearPoll(): void {
    if (this.#pollTimer) clearTimeout(this.#pollTimer);
    this.#pollTimer = null;
  }

  #armPoll(): void {
    this.#clearPoll();
    const after = this.local?.poll_after_s;
    if (this.#stopped || after == null) return;
    this.#pollTimer = setTimeout(() => {
      this.#pollTimer = null;
      void this.pollLocal();
    }, after * 1000);
  }

  async loadModelChoices(): Promise<void> {
    try {
      const v = await (this.#deps.getConfigVocabulary ?? getConfigVocabulary)(false);
      this.modelChoices = v.model_choices ?? [];
      this.modelChoiceLabels = v.labels?.scope ?? {};
      this.modelChoicesError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.modelChoicesError = apiErrorText(e);
    }
  }

  async deactivate(): Promise<boolean> {
    const ok = await this.active.deactivate();
    if (ok) await this.loadRegistry();
    return ok;
  }

  async probe(name: string): Promise<void> {
    if (this.probing) return;
    this.probing = name;
    const { [name]: _drop, ...rest } = this.probeErrors;
    void _drop;
    this.probeErrors = rest;
    try {
      this.probes = {
        ...this.probes,
        [name]: await (this.#deps.probeVlmEndpoint ?? probeVlmEndpoint)(name),
      };
      await this.loadRegistry();
    } catch (e) {
      this.probeErrors = { ...this.probeErrors, [name]: apiErrorText(e) };
    } finally {
      this.probing = null;
    }
  }

  /** Records the wish to serve `catalogId`; `force` is sent only for a
   *  retry after a served `vlm_catalog_does_not_fit`. */
  async selectLocal(catalogId: string, force: boolean): Promise<boolean> {
    if (this.localBusy) return false;
    this.localBusy = true;
    this.localError = null;
    this.localErrorCode = null;
    try {
      const local = await (this.#deps.selectLocalVlm ?? selectLocalVlm)({
        catalog_id: catalogId,
        ...(force ? { force: true } : {}),
      });
      if (this.catalog) this.catalog = { ...this.catalog, local };
      this.#armPoll();
      await this.loadCatalog();
      return true;
    } catch (e) {
      this.localError = apiErrorText(e);
      this.localErrorCode = configErrorDetail(e)?.error ?? null;
      return false;
    } finally {
      this.localBusy = false;
    }
  }

  /** Cancels the pending switch request. */
  async clearLocal(): Promise<boolean> {
    if (this.localBusy) return false;
    this.localBusy = true;
    this.localError = null;
    this.localErrorCode = null;
    try {
      const local = await (this.#deps.clearLocalVlmSelection ?? clearLocalVlmSelection)();
      if (this.catalog) this.catalog = { ...this.catalog, local };
      this.#armPoll();
      await this.loadCatalog();
      return true;
    } catch (e) {
      this.localError = apiErrorText(e);
      this.localErrorCode = configErrorDetail(e)?.error ?? null;
      return false;
    } finally {
      this.localBusy = false;
    }
  }
}

export function createVlmModels(deps: Partial<VlmModelsDeps> = {}): VlmModels {
  return new VlmModels(deps);
}
