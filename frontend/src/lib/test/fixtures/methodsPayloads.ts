// Two fixtures backing the mock-backed verification for the whole scoped-
// assist feature (docs/design/vlm-scoped-labeling-assist-plan-2026-09-20.md
// §6). Kept in sync by eye with the JSON stubbed in
// scripts/playwright_assist_scope.py.

/** Today's real `/methods` shape — no assist axes at all. */
export const METHODS_TODAY = {
  strategies: [
    {
      id: 'ivf',
      axis: 'cluster',
      label: 'FAISS IVF-512 (production)',
      status: 'stable',
      default: true,
    },
    {
      id: 'default',
      axis: 'sort',
      label: 'Recent first',
      status: 'stable',
      default: true,
    },
    {
      id: 'representativeness',
      axis: 'sort',
      label: 'Representativeness',
      status: 'stable',
    },
    {
      id: 'yolo',
      axis: 'export',
      label: 'YOLO detection dataset export',
      status: 'stable',
      default: true,
    },
  ],
  flags: {},
};

/** Same payload as `METHODS_TODAY` plus the two assist axes — one usable
 *  entry each, plus one deliberately-shadow profile and one
 *  deliberately-disabled pack to exercise the status filter. */
export const METHODS_WITH_ASSIST_AXES = {
  strategies: [
    ...METHODS_TODAY.strategies,
    {
      id: 'grounding_v2',
      axis: 'detection_profile',
      label: 'Grounding detector v2',
      status: 'stable',
      default: true,
    },
    {
      id: 'sam3_dense',
      axis: 'detection_profile',
      label: 'SAM3 dense proposals',
      status: 'experimental',
    },
    {
      id: 'legacy_profile',
      axis: 'detection_profile',
      label: 'Legacy profile (mid-validation)',
      status: 'shadow',
    },
    {
      id: 'warehouse_v1',
      axis: 'prompt_pack',
      label: 'Warehouse vocabulary',
      status: 'stable',
      default: true,
    },
    {
      id: 'retired_pack',
      axis: 'prompt_pack',
      label: 'Retired prompt pack',
      status: 'disabled',
    },
  ],
  flags: {},
};
