/**
 * One prompt pack's editor state (any_domain_plan.md §3.3, §3.5, §7.2,
 * §7.6 items 1 and 4; docs/design/w3-pack-editor-ui-plan-2026-09-27.md §3):
 * the pack binding of the shared `ConfigEditor` (draft, served live
 * validation, Save with `expected_revision` and its conflict paths,
 * revisions and restore, the `config.changed` wake-up).
 */
import {
  getPromptPack,
  getPromptPackRevision,
  getPromptPackRevisions,
  getPromptPackSchema,
  updatePromptPack,
  validatePromptPack,
} from '$lib/api';
import { ConfigEditor } from '$lib/config/configEditor.svelte';
import { isConfigAxisEvent } from '$lib/config/validationIssues';
import { subscribeCurationEvents, type CurationEvent } from '$lib/sse';
import type { PromptPackBody, PromptPackDoc, PromptPackSchema } from '$lib/types_packs';
import { PackActive, type PackActiveDeps } from './packActive.svelte';

export interface PackEditorDeps extends PackActiveDeps {
  getPromptPack: typeof getPromptPack;
  getPromptPackSchema: typeof getPromptPackSchema;
  getPromptPackRevisions: typeof getPromptPackRevisions;
  getPromptPackRevision: typeof getPromptPackRevision;
  updatePromptPack: typeof updatePromptPack;
  validatePromptPack: typeof validatePromptPack;
  subscribe: (onEvent: (e: CurationEvent) => void) => { close(): void };
}

export class PackEditor extends ConfigEditor<
  PromptPackBody,
  PromptPackDoc,
  PromptPackSchema,
  PackActive
> {
  constructor(name: string, deps: Partial<PackEditorDeps> = {}) {
    super(
      name,
      {
        getSchema: () => (deps.getPromptPackSchema ?? getPromptPackSchema)(),
        getDoc: (n) => (deps.getPromptPack ?? getPromptPack)(n),
        getRevisions: (n) => (deps.getPromptPackRevisions ?? getPromptPackRevisions)(n),
        getRevision: (n, r) =>
          (deps.getPromptPackRevision ?? getPromptPackRevision)(n, r),
        update: (n, body) => (deps.updatePromptPack ?? updatePromptPack)(n, body),
        validate: (body, signal) =>
          (deps.validatePromptPack ?? validatePromptPack)(body, signal),
        subscribe:
          deps.subscribe ??
          ((onEvent) => subscribeCurationEvents({ topic: 'config', onEvent })),
        isEvent: (e) => isConfigAxisEvent(e, 'prompt_pack'),
      },
      new PackActive(deps),
    );
  }
}

export function createPackEditor(
  name: string,
  deps: Partial<PackEditorDeps> = {},
): PackEditor {
  return new PackEditor(name, deps);
}
