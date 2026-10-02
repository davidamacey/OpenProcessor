/**
 * The project's active VLM endpoint (`GET {scoped}/vlm/endpoints/active`,
 * axis `vlm`) and the writes that move it: activate, rollback and
 * deactivate ("VLM off for this project"). The shared `ConfigActive`
 * (any_domain_plan.md §7.8; docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md
 * §3.2) plus
 *
 * - `acknowledge_external` on activate (via `extra`),
 * - `ackRequired`: the served 422 `vlm_external_not_acknowledged` detail
 *   (which endpoint, and where to acknowledge it),
 * - after every successful write: `/health` re-polled and `/methods`
 *   dropped, since the per-run ack flags and the VLM status change.
 */
import { activateVlm, deactivateVlm, getActiveVlm, rollbackVlm } from '$lib/api_vlm';
import { ConfigActive } from '$lib/config/configActive.svelte';
import { healthStore } from '$stores/health.svelte';
import { strategiesStore } from '$stores/strategies.svelte';
import type { VlmActiveResponse } from '$lib/types_vlm';

export interface VlmActiveDeps {
  getActiveVlm: typeof getActiveVlm;
  activateVlm: typeof activateVlm;
  rollbackVlm: typeof rollbackVlm;
  deactivateVlm: typeof deactivateVlm;
  /** After a successful write (default: re-poll `/health`, reset `/methods`). */
  onchanged: () => void;
}

export interface AckRequired {
  endpoint: string | null;
  activateVia: string | null;
  message: string;
}

export class VlmActive extends ConfigActive<VlmActiveResponse> {
  constructor(deps: Partial<VlmActiveDeps> = {}) {
    super({
      getActive: () => (deps.getActiveVlm ?? getActiveVlm)(),
      activate: (name, body) =>
        (deps.activateVlm ?? activateVlm)(name, {
          revision: body.revision,
          expected_active: body.expected_active,
          force: body.force,
          ...(body.acknowledge_external === true ? { acknowledge_external: true } : {}),
        }),
      rollback: (body) => (deps.rollbackVlm ?? rollbackVlm)(body),
      deactivate: (body) => (deps.deactivateVlm ?? deactivateVlm)(body),
    });
    this.onchanged =
      deps.onchanged ??
      (() => {
        void healthStore.poll();
        strategiesStore.reset();
      });
  }

  /** The served refusal when an external endpoint needs an
   *  acknowledgement, else null. */
  get ackRequired(): AckRequired | null {
    const d = this.errorDetail;
    if (d?.error !== 'vlm_external_not_acknowledged') return null;
    return {
      endpoint: d.endpoint ?? null,
      activateVia: d.activate_via ?? null,
      message: d.message,
    };
  }
}
