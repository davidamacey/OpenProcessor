/**
 * Wire types for the selectable assist runs (`{prefix}/assist/runs`), field for
 * field from the vendored contract (`contracts/openapi/curation.json`) and
 * pinned by `contract/assistContract.test.ts`. Hand-written like the other
 * `types_*.ts`: the backend serves every figure and decision here
 * (sample sizes, call estimates, the proposal and its gate, the stale-scope
 * check); nothing in the app computes them.
 */
import type { ItemFilter } from './types_itemFilter';

export const ASSIST_ACTIONS = ['classify', 'attributes', 'verify', 'segment'] as const;
export type AssistAction = (typeof ASSIST_ACTIONS)[number];

export const ASSIST_STATUSES = [
  'dry_run',
  'queued',
  'running',
  'waiting_region',
  'completed',
  'completed_with_errors',
  'failed',
  'cancelled',
  'interrupted',
] as const;
export type AssistStatus = (typeof ASSIST_STATUSES)[number];

export type AssistConfidence = 'high' | 'medium' | 'low';

export interface AssistContext {
  cluster_id: number;
  cluster_subid?: string | null;
}

export interface BoxRef {
  crop_id: string;
  box_id: string;
}

export interface AssistCropsTarget {
  kind: 'crops';
  crop_ids: string[];
  context?: AssistContext | null;
}

export interface AssistBoxesTarget {
  kind: 'boxes';
  boxes: BoxRef[];
}

export interface AssistClusterTarget {
  kind: 'cluster';
  cluster_id: number;
  cluster_subid?: string | null;
}

export interface AssistFilterTarget {
  kind: 'filter';
  filter: ItemFilter;
}

export type AssistTarget =
  AssistCropsTarget | AssistBoxesTarget | AssistClusterTarget | AssistFilterTarget;

export type AssistSampleMode = 'all' | 'representative';
export type AssistStratifyBy = 'auto' | 'cluster_subid' | 'none';

export interface AssistSample {
  mode?: AssistSampleMode;
  size?: number | null;
  seed?: number;
  stratify_by?: AssistStratifyBy;
}

export interface AssistSegmentSpec {
  verify: 'vlm' | 'none';
}

export interface AssistRunRequest {
  target: AssistTarget;
  actions: AssistAction[];
  sample?: AssistSample;
  vlm?: string | null;
  prompt_pack?: string | null;
  acknowledge_external?: boolean;
  segment?: AssistSegmentSpec | null;
  min_agreement?: number;
  dry_run?: boolean;
}

export interface AssistStratum {
  key: string;
  size: number;
  sampled: number;
}

export interface AssistSkipped {
  holdout?: number;
  excluded?: number;
  class_locked?: number;
  box_locked?: number;
  box_not_found?: number;
  not_found?: number;
}

/** The endpoint or pack a run pinned (`{name, revision, model?}`). */
export type AssistPinned = Record<string, unknown>;

export interface AssistResolved {
  scope_count: number;
  scope_digest: string;
  sampled: number;
  sample_seed: number;
  strata?: AssistStratum[];
  skipped?: AssistSkipped;
  vlm?: AssistPinned | null;
  prompt_pack?: AssistPinned | null;
  estimated_vlm_calls?: number;
}

export interface AssistProgress {
  processed?: number;
  total?: number;
}

export interface AssistClassifyResult {
  predicted?: number;
  updated?: number;
  failed?: number;
  unreadable?: number;
  locked_suggest_only?: number;
}

export interface AssistAttributesResult {
  items_written?: number;
  invalid_values?: number;
}

export interface AssistVerifyResult {
  boxes?: number;
  accepted?: number;
  rejected?: number;
  locked_skipped?: number;
}

export interface AssistSegmentResult {
  queued?: number;
  done?: number;
  pending?: number;
  failed?: number;
}

export interface ClassVotes {
  class_name: string;
  votes: number;
}

export interface NewClassVotes {
  name: string;
  votes: number;
}

export interface AttributeValueCount {
  value: string;
  count: number;
}

export interface AttributeSummary {
  key: string;
  values: AttributeValueCount[];
  answered: number;
}

export interface AssistProposal {
  class_name: string | null;
  in_registry: boolean;
  agreement: number;
  answered: number;
  min_agreement: number;
  passes_gate: boolean;
  distribution: ClassVotes[];
  new_class_proposals: NewClassVotes[];
  attribute_summary: AttributeSummary[];
}

export interface AssistResult {
  classify?: AssistClassifyResult | null;
  attributes?: AssistAttributesResult | null;
  verify?: AssistVerifyResult | null;
  segment?: AssistSegmentResult | null;
  proposal?: AssistProposal | null;
}

export interface AssistScopeNow {
  count: number;
  overlap: number;
  jaccard: number;
  changed: boolean;
}

export interface AssistRunState {
  run_id?: string | null;
  status: AssistStatus;
  created_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
  request: AssistRunRequest;
  resolved: AssistResolved;
  progress?: AssistProgress;
  result?: AssistResult | null;
  scope_now?: AssistScopeNow | null;
}

export interface AssistRunList {
  runs: AssistRunState[];
}

export interface AssistApplyRequest {
  class_name: string;
  label_source?: 'human';
  allow_changed_scope?: boolean;
  dry_run?: boolean;
}

export interface AssistApplyConflict {
  crop_id: string;
  current_source?: string | null;
}

export interface AssistApplyResponse {
  dry_run: boolean;
  class_name: string;
  selected: number;
  skipped_locked: number;
  updated?: number;
  updated_ids?: string[];
  conflicts?: AssistApplyConflict[];
  scope_now: AssistScopeNow;
}

/** One suggested attribute on an item or a box. Never a label. */
export interface AssistAttribute {
  key: string;
  value: string;
  confidence: AssistConfidence;
}

export interface AssistBoxAttributes {
  box_id: string;
  attributes?: AssistAttribute[];
}

/** The newest run's suggestions for an item (`ItemAssistWire`). */
export interface ItemAssist {
  run_id: string;
  run_created_at: string;
  vlm_endpoint?: string | null;
  vlm_model?: string | null;
  prompt_pack?: string | null;
  prompt_pack_revision?: number | null;
  class_name?: string | null;
  class_confidence?: AssistConfidence | null;
  attributes?: AssistAttribute[];
  boxes?: AssistBoxAttributes[];
}
