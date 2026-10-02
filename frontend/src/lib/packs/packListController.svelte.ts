/**
 * The prompt-pack list page's state (any_domain_plan.md §7.6 items 1 and
 * 4; docs/design/w3-pack-editor-ui-plan-2026-09-27.md §2): the pack
 * binding of the shared `ConfigList` (the served list, the active pack
 * with Rollback, Clone and Delete; `config.changed axis=prompt_pack` is a
 * wake-up to re-read).
 */
import { clonePromptPack, deletePromptPack, listPromptPacks } from '$lib/api';
import { ConfigList } from '$lib/config/configList.svelte';
import { isConfigAxisEvent } from '$lib/config/validationIssues';
import { subscribeCurationEvents, type CurationEvent } from '$lib/sse';
import type { PromptPackDoc, PromptPackList } from '$lib/types_packs';
import { PackActive, type PackActiveDeps } from './packActive.svelte';

export interface PackListDeps extends PackActiveDeps {
  listPromptPacks: typeof listPromptPacks;
  clonePromptPack: typeof clonePromptPack;
  deletePromptPack: typeof deletePromptPack;
  subscribe: (onEvent: (e: CurationEvent) => void) => { close(): void };
}

export class PackList extends ConfigList<PromptPackList, PromptPackDoc, PackActive> {
  constructor(deps: Partial<PackListDeps> = {}) {
    super(
      {
        list: () => (deps.listPromptPacks ?? listPromptPacks)(),
        clone: (name, body) => (deps.clonePromptPack ?? clonePromptPack)(name, body),
        remove: (name, rev) => (deps.deletePromptPack ?? deletePromptPack)(name, rev),
        subscribe:
          deps.subscribe ??
          ((onEvent) => subscribeCurationEvents({ topic: 'config', onEvent })),
        isEvent: (e) => isConfigAxisEvent(e, 'prompt_pack'),
      },
      new PackActive(deps),
    );
  }
}

export function createPackList(deps: Partial<PackListDeps> = {}): PackList {
  return new PackList(deps);
}
