/**
 * Where an "all model choices" row can be changed (question A-5: the
 * backend serves no editor hint per role, so this is a four-entry map
 * keyed by the served `role`). A role with no entry renders without a
 * link. Paths are project-relative; the caller wraps them in
 * `projectHref`.
 */
export const MODEL_CHOICE_LINKS: Record<string, `/settings${string}`> = {
  vlm: '/settings/models#endpoints',
  local_vlm_model: '/settings/models#local-model',
  region_detector: '/settings/region-profiles',
  region_ocr: '/settings/region-profiles',
};

export function modelChoiceLink(role: string): `/settings${string}` | null {
  return MODEL_CHOICE_LINKS[role] ?? null;
}
