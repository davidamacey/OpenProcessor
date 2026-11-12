/**
 * What is active for one preset kind and the write that moves it. A project kind is active per axis
 * (`GET /active?axis=`, a cluster profile has two); the serving profile is one deployment-wide
 * setting. Every write sends what was last read as its compare-and-set token (`expected_active`, or
 * the serving `setting_revision`); a served 409 re-reads so the next attempt starts from what is
 * really active.
 */
import {
  activatePreset,
  activateServingProfile,
  getActivePreset,
  getActiveServingProfile,
  getPresetSchema,
  presetRefusalText,
} from '$lib/api_presets';
import { configErrorDetail } from '$lib/api_profiles';
import type { ConfigErrorDetail } from '$lib/types_config';
import type { ActivePreset, PresetSchemaWire } from '$lib/types_presets';
import type { PresetKind } from './presetKinds';

export interface PresetActiveDeps {
  getActivePreset: typeof getActivePreset;
  getActiveServingProfile: typeof getActiveServingProfile;
  activatePreset: typeof activatePreset;
  activateServingProfile: typeof activateServingProfile;
  getPresetSchema: typeof getPresetSchema;
}

export class PresetActive {
  /** One entry per axis, in the kind's axis order; one for the serving profile. */
  entries = $state<ActivePreset[]>([]);
  loadError = $state<string | null>(null);
  /** The JSON Schema of the kind's body, as served; read once. */
  schema = $state<PresetSchemaWire | null>(null);
  schemaError = $state<string | null>(null);
  busy = $state(false);
  actionError = $state<string | null>(null);
  errorDetail = $state<ConfigErrorDetail | null>(null);
  readonly kind: PresetKind;
  #deps: Partial<PresetActiveDeps>;

  constructor(kind: PresetKind, deps: Partial<PresetActiveDeps> = {}) {
    this.kind = kind;
    this.#deps = deps;
  }

  async load(): Promise<void> {
    await Promise.all([this.#loadSchema(), this.#loadActive()]);
  }

  async #loadSchema(): Promise<void> {
    if (this.schema) return;
    try {
      this.schema = await (this.#deps.getPresetSchema ?? getPresetSchema)(this.kind.id);
      this.schemaError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.schemaError = presetRefusalText(e);
    }
  }

  async #loadActive(): Promise<void> {
    const d = this.#deps;
    const { id, axes } = this.kind;
    try {
      if (id === 'serving_profile') {
        this.entries = [await (d.getActiveServingProfile ?? getActiveServingProfile)()];
      } else {
        const read = d.getActivePreset ?? getActivePreset;
        this.entries = await Promise.all(axes.map((a) => read(id, a.id)));
      }
      this.loadError = null;
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return;
      this.loadError = presetRefusalText(e);
    }
  }

  clearAction(): void {
    this.actionError = null;
    this.errorDetail = null;
  }

  /** The axes (their ids; `null` for serving) on which `name` is the active document. */
  activeOn(name: string): (string | null)[] {
    return this.entries
      .filter((e) => e.state === 'active' && e.name === name)
      .map((e) => e.axis);
  }

  /** Makes `name` at `revision` the active one on `axis` (`null` for the serving profile). */
  async activate(name: string, revision: number, axis: string | null): Promise<boolean> {
    const current = this.entries.find((e) => e.axis === axis);
    if (!current || this.busy) return false;
    this.busy = true;
    this.clearAction();
    const d = this.#deps;
    try {
      if (this.kind.id === 'serving_profile') {
        await (d.activateServingProfile ?? activateServingProfile)(name, {
          revision,
          expected_revision: current.setting_revision,
        });
      } else {
        await (d.activatePreset ?? activatePreset)(this.kind.id, name, {
          axis,
          revision,
          expected_active:
            current.state === 'active'
              ? { name: current.name, revision: current.revision }
              : null,
        });
      }
      await this.load();
      return true;
    } catch (e) {
      this.actionError = presetRefusalText(e);
      const detail = configErrorDetail(e);
      this.errorDetail = detail;
      if (detail?.error === 'active_conflict' || detail?.error === 'revision_conflict') {
        await this.load();
      }
      return false;
    } finally {
      this.busy = false;
    }
  }
}
