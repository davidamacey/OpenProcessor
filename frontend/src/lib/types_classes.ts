/**
 * Class-registry wire types: registry entries, thresholds, create / update /
 * merge bodies and the merge dry run.
 */
export interface RegistryClass {
  id: number;
  name: string;
  group: string | null;
  count: number;
  validated_count: number;
  /**
   * Size of the FAISS cluster bucket whose id == this class id. Includes
   * unlabeled candidates the operator hasn't triaged yet — i.e. matches
   * what the operator sees when they open /clusters/{id}. The sidebar
   * chip displays this; the cluster-page banner breaks out validated /
   * labeled / total side-by-side.
   */
  cluster_size: number;
  /** Single-character keyboard shortcut, persisted in the registry. */
  hotkey_letter?: string | null;
  added_at: string;
  /** Hex color hint or null. */
  color?: string | null;
  /** True when the class has been merged into another and should be hidden by default. */
  deprecated?: boolean;
  /** OpenProcessor 51b05d7: the class this one was merged into (its crops
   *  now carry that class). Null/absent for a class never merged. A merged
   *  class can't be restored (restore 409s `class_merged`). */
  merged_into?: number | null;
  /** Server-computed adequacy tier from `GET {API_PREFIX}/classes`
   *  (`block` | `warn` | `ok`, against the served `thresholds`). Never
   *  recomputed client-side from `validated_count`. */
  adequacy: string;
  /** `'item'` (an ordinary item class) or `'region'` (a slot-bound
   *  region class, i.e. the served profile's region_class_name) — served on
   *  `GET {API_PREFIX}/classes`. The single source of truth for excluding
   *  a region class from an item-class picker (`isSlotBoundClass`/
   *  `isItemClassTarget`, `$lib/classVisibility`). */
  kind: 'item' | 'region';
  /** Validated crops usable for training (region counts, test holdout and
   *  excluded crops left out) — the server's own count, not client math. */
  trainable: number;
  /** The shortfall of `trainable` against the served per-class hard
   *  minimum (`thresholds.block_below`), floored at 0. > 0 means the
   *  class blocks preflight. */
  trainable_gap: number;
}

/** `thresholds` served on `GET {API_PREFIX}/classes`, `GET {API_PREFIX}/stats/classes`
 *  and `{API_PREFIX}/train/preflight` — the single source of truth for the
 *  adequacy tiers, augmentation target range and test-holdout minimum. */
export interface ClassThresholds {
  block_below: number;
  warn_below: number;
  min_test_per_class: number;
  aug_target_min: number;
  aug_target_max: number;
}

/** `GET {API_PREFIX}/classes` response envelope. */
export interface ClassesResponse {
  classes: RegistryClass[];
  thresholds: ClassThresholds;
  /** Every single-character combo reserved for a labeling action —
   *  core keys plus every registered queue-capable slot's keymap. */
  reserved_hotkeys: string[];
}

/** Payload for `POST {API_PREFIX}/classes`. */
export interface RegistryClassCreate {
  name: string;
  group: string;
  notes?: string;
}

/** Payload for `PUT {API_PREFIX}/classes/{id}`. Any subset of fields may be supplied. */
export interface RegistryClassUpdate {
  name?: string;
  group?: string;
  /** Pass an empty string to clear the binding, or omit to leave unchanged. */
  hotkey_letter?: string;
}

/** Payload for `POST {API_PREFIX}/classes/merge`. */
export interface RegistryClassMerge {
  source_id: number;
  target_id: number;
}

/** `POST {API_PREFIX}/classes/merge?dry_run=true` response — reports
 *  counts and writes nothing. A real merge 409s when `holdout_blocking > 0`. */
export interface ClassMergeDryRun {
  dry_run: true;
  source_id: number;
  target_id: number;
  would_relabel: number;
  /** OpenProcessor 51b05d7: human validations KEPT on the relabeled crops
   *  (a merge now carries validations over; was `would_unvalidate`). */
  validations_carried_over: number;
  holdout_blocking: number;
  blocked: boolean;
}
