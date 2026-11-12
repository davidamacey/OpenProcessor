import type { EmbeddingState } from '$lib/types_itemFilter';

/**
 * The item fields the raw wire (`RawCrop`) and the labeler's `Crop` carry alike: dataset
 * import, combine and embedding provenance. Each is served on every item; a missing key maps
 * to null / false / [].
 */
export type ItemProvenanceFields = {
  /** a human-locked label the pipeline must not overwrite. */
  label_locked?: boolean;
  /** ids of the dataset imports that carried this item. */
  import_ids?: string[];
  dataset_split?: string | null;
  imported_at?: string | null;
  proposed_by_import?: string | null;
  on_negative_frame?: boolean;
  import_standalone_region?: boolean;
  proposal_chain?: string[];
  /** where a combined item came from (null on a non-combined one). */
  origin_project?: string | null;
  origin_item_id?: string | null;
  origin_image_id?: string | null;
  origin_split?: string | null;
  combine_conflict?: boolean;
  combine_conflict_origins?: string[];
  combine_merged_origins?: string[];
  /** Per embedding space: whether the item has a vector in it and why not. A
   *  space with no state for the item (written before the field existed, or
   *  not built yet) has no key. */
  embedding_states?: Record<string, EmbeddingState>;
  /** The member count of the item's near-duplicate group, and whether the item represents it.
   *  Null for an item in no group. */
  near_dup_group_size?: number | null;
  near_dup_is_representative?: boolean | null;
};
