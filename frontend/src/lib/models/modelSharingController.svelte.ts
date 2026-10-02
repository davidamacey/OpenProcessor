/**
 * `/models` sharing state and writes (projects P2, §5.5), in the same
 * factory convention as the other controllers: the page owns the model
 * list and hands its reload in; this owns the sharing write and the
 * lazily-loaded class mappings.
 *
 * Thin frontend: the write sends the model's served sharing revision as
 * `expected_revision`, and every refusal is the served
 * `{detail: {error, message}}` — `message` verbatim, `error` only to pick
 * the follow-up (`revision_conflict` reloads the list so the next attempt
 * carries the fresh served revision; `in_use` offers the served `force`).
 */
import {
  getModelClassMapping,
  projectErrorDetail,
  projectErrorText,
  setModelSharing,
} from '$lib/api';
import type { ModelInfo } from '$lib/types';
import type {
  ModelClassMappingResponse,
  ModelSharingResponse,
  ModelSharingUser,
} from '$lib/types_models';

export type SharingResult =
  | { ok: true; response: ModelSharingResponse }
  | {
      ok: false;
      /** The served `detail.error`, or `null` for an unstructured error. */
      code: string | null;
      message: string;
      /** The served `detail.used_by` on a 409 `in_use`. */
      usedBy: ModelSharingUser[];
    };

export type MappingState =
  | { status: 'loading' }
  | { status: 'ok'; mapping: ModelClassMappingResponse }
  | { status: 'error'; message: string };

export function createModelSharing(opts: { reload: () => Promise<void> }) {
  let pending = $state<string | null>(null);
  const mappings = $state<Record<string, MappingState>>({});

  return {
    /** The model name mid-write, or `null`. */
    get pending() {
      return pending;
    },
    mapping(name: string): MappingState | undefined {
      return mappings[name];
    },
    /**
     * Flip the served `shared` flag. Callers only offer this when the
     * served `sharing_revision` exists (`canToggleSharing`).
     */
    async toggle(m: ModelInfo, force = false): Promise<SharingResult> {
      if (typeof m.sharing_revision !== 'number') {
        return {
          ok: false,
          code: null,
          message: 'The server did not send this model’s sharing revision.',
          usedBy: [],
        };
      }
      pending = m.name;
      try {
        const response = await setModelSharing(
          m.name,
          { shared: !m.shared, expected_revision: m.sharing_revision },
          force,
        );
        await opts.reload();
        return { ok: true, response };
      } catch (e) {
        const detail = projectErrorDetail(e);
        if (detail?.error === 'revision_conflict') await opts.reload();
        return {
          ok: false,
          code: detail?.error ?? null,
          message: projectErrorText(e),
          usedBy: detail?.used_by ?? [],
        };
      } finally {
        pending = null;
      }
    },
    /** Load `GET {scoped}/models/{name}/class_mapping` once per model. */
    async loadMapping(name: string): Promise<void> {
      const cur = mappings[name];
      if (cur && cur.status !== 'error') return;
      mappings[name] = { status: 'loading' };
      try {
        mappings[name] = { status: 'ok', mapping: await getModelClassMapping(name) };
      } catch (e) {
        if ((e as Error)?.name === 'AbortError') {
          delete mappings[name];
          return;
        }
        mappings[name] = { status: 'error', message: projectErrorText(e) };
      }
    },
  };
}

export type ModelSharing = ReturnType<typeof createModelSharing>;
