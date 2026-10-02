/**
 * The VLM picker's choice, as the test routes take it
 * (`POST /prompt_packs/test`, `POST /region_profiles/test`: `vlm_name`,
 * `vlm_revision`, `acknowledge_external`). A test names a registry
 * endpoint by name and never a revision (`vlm_revision: null` = the
 * endpoint's current one); the unsaved-draft option (`vlm_draft`) is not
 * offered. Nothing is sent for the default pick, and
 * `acknowledge_external` only when the operator ticked it.
 */
import type { TestVlmSelection } from '$lib/types_configTest';

export interface TestVlmPick {
  vlm: string | null;
  acknowledgeExternal: boolean;
}

export function toTestVlmSelection(pick: TestVlmPick): TestVlmSelection | null {
  if (pick.vlm == null) return null;
  const sel: TestVlmSelection = { vlm_name: pick.vlm, vlm_revision: null };
  if (pick.acknowledgeExternal) sel.acknowledge_external = true;
  return sel;
}

export function fromTestVlmSelection(sel: TestVlmSelection | null): TestVlmPick {
  return {
    vlm: sel?.vlm_name ?? null,
    acknowledgeExternal: sel?.acknowledge_external === true,
  };
}
