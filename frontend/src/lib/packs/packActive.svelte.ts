/**
 * The active prompt pack of the current project (`GET /prompt_packs/active`)
 * and the two writes that move it: activate and rollback
 * (any_domain_plan.md §7.2, §7.6 item 4; docs/design/
 * w3-pack-editor-ui-plan-2026-09-27.md §2, §3). The pack binding of the
 * shared `ConfigActive`: packs have no deactivate route.
 */
import { activatePromptPack, getActivePromptPack, rollbackPromptPack } from '$lib/api';
import { ConfigActive } from '$lib/config/configActive.svelte';
import type { ActivateResponse } from '$lib/types_config';

export interface PackActiveDeps {
  getActivePromptPack: typeof getActivePromptPack;
  activatePromptPack: typeof activatePromptPack;
  rollbackPromptPack: typeof rollbackPromptPack;
}

export class PackActive extends ConfigActive<ActivateResponse> {
  constructor(deps: Partial<PackActiveDeps> = {}) {
    super({
      getActive: () => (deps.getActivePromptPack ?? getActivePromptPack)(),
      activate: (name, body) =>
        (deps.activatePromptPack ?? activatePromptPack)(name, body),
      rollback: (body) => (deps.rollbackPromptPack ?? rollbackPromptPack)(body),
    });
  }
}
