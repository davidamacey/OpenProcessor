/**
 * Wire types for OpenProcessor W4 (region-profile CRUD and the config
 * vocabulary): `any_domain_plan.md` §4, §7.3, §7.4; `projects_plan.md`
 * §5.5 (shared detectors). Every route is project-scoped
 * (`{prefix}/region_profiles...`, `{prefix}/config/vocabulary`).
 *
 * Built against the frozen spec before the backend lands W4, so nothing
 * here is pinned to the vendored OpenAPI yet. Where the spec leaves a
 * shape open, the type is the literal reading recorded in
 * docs/design/w4-profile-editor-ui-plan-2026-09-27.md §8 (question W4-Qn).
 */

import type {
  ActiveConfigResponse,
  ActiveRef,
  ConfigDocBase,
  ConfigSource,
  ConfigUpdateRequest,
  ConfigValidateRequest,
  ValidationReport,
} from './types_config';
import type { ReprocessRequest } from './types_import';

/** A value of one profile field, JSON-typed (§7.3: tuples -> arrays,
 *  frozensets -> sorted arrays). */
export type ProfileFieldValue =
  string | number | boolean | null | string[] | number[] | Record<string, unknown>;

/**
 * A profile body: every `DetectionProfile` field except `name`, keyed by
 * the schema's `field` ids. Kept as a record so a field the backend adds
 * renders from the schema alone.
 */
export type RegionProfileBody = Record<string, ProfileFieldValue>;

/** One `GET /region_profiles` `profiles[]` row (§7.3). */
export interface RegionProfileSummary {
  name: string;
  source: ConfigSource;
  read_only: boolean;
  revision: number | null;
  etag: string;
  display_name: string | null;
  display_name_singular?: string | null;
  region_class_name: string | null;
  text_reader: string | null;
  reads_text: boolean;
  detector_model: string | null;
  segmenter_text_prompt: string | null;
  parent_classes: string[];
  max_regions_per_item: number | null;
  active: boolean;
  /** Present when `active`: the revision that is active (§4.4 "saved r4,
   *  active r3"). */
  active_revision?: number | null;
  updated_at: string | null;
}

/** One `templates[]` row: clone-only, never activatable (§4.1). */
export interface RegionProfileTemplate {
  name: string;
  source: ConfigSource;
  read_only: boolean;
  path: string;
  display_name: string | null;
  reads_text: boolean;
}

export interface RegionProfileList {
  profiles: RegionProfileSummary[];
  /** Served only with `include_templates=true`. */
  templates?: RegionProfileTemplate[];
  active: ActiveRef;
  config_revision: number;
  stale: boolean;
}

/** `effective` on a profile doc: what the saved revision does. */
export interface RegionProfileEffective {
  reads_text: boolean;
  text_hint_active: boolean;
  legs: string[];
  segmenter_enabled: boolean;
}

/** `GET /region_profiles/{name}` and `.../revisions/{rev}`. */
export interface RegionProfileDoc extends ConfigDocBase<RegionProfileBody> {
  effective?: RegionProfileEffective | null;
}

/** A schema row's value type (§7.3). */
export type ProfileFieldType =
  | 'string'
  | 'int'
  | 'float'
  | 'bool'
  | 'enum'
  | 'string_list'
  | 'int_list'
  | 'float_pair'
  | 'rgb'
  | (string & {});

/** The §7.3 `choices_from` lists. */
export type ChoicesFrom =
  | 'detectors'
  | 'segmenters'
  | 'ocr_pipeline_models'
  | 'ocr_det_models'
  | 'ocr_rec_models'
  | 'registry_classes'
  | 'text_reader_modes'
  | 'vlm_catalog'
  | 'secret_refs'
  | (string & {});

/** The uniform `{id, label}` every vocabulary list entry carries: store
 *  `id`, show `label`. */
export interface Choice {
  id: string;
  label: string;
}

/** `applies_when` (§7.3): the GUI dims a field whose leg / mode is off. */
export type AppliesWhen =
  'detector' | 'segmenter' | 'reads_text' | 'text_hint' | (string & {});

/** One `GET /region_profiles/schema` `fields[]` row (§7.3). */
export interface ProfileSchemaField {
  field: string;
  label: string;
  group: string;
  type: ProfileFieldType;
  default: ProfileFieldValue;
  min: number | null;
  max: number | null;
  enum: Choice[] | null;
  advanced: boolean;
  applies_when: AppliesWhen | null;
  choices_from?: ChoicesFrom | null;
  /** What the "none" option means for this field: its stored id (`""`)
   *  and label. `null` = no empty choice. */
  empty_choice?: { id: string | null; label: string } | null;
  help: string;
}

export interface ProfileSchemaGroup {
  id: string;
  label: string;
}

export interface RegionProfileSchema {
  fields: ProfileSchemaField[];
  groups: ProfileSchemaGroup[];
}

export type ProfileUpdateRequest = ConfigUpdateRequest<RegionProfileBody>;
export type ProfileValidateRequest = ConfigValidateRequest<RegionProfileBody>;

/** One `by_profile[]` row of `ActivationImpact`. */
export interface ImpactByProfile {
  name: string | null;
  revision: number | null;
  count: number;
}

/** `GET /region_profiles/active/impact` (§4.6, §7.3). `suggested_reprocess`
 *  is W10's request (G2: absent when W4 ships first). */
export interface ActivationImpact {
  items_total: number;
  by_profile: ImpactByProfile[];
  validated_items: number;
  unseeded_items: number;
  pending_items: number;
  pending_not_matching: number;
  suggested_reprocess?: ReprocessRequest | null;
}

/** `POST /region_profiles/{name}/activate` → 200 (§7.3). `impact` and
 *  `validation` are optional: W4-Q6. */
export interface ProfileActivateResponse extends ActiveConfigResponse {
  impact?: ActivationImpact | null;
  validation?: ValidationReport | null;
}

// -- GET /config/vocabulary (§7.4) ------------------------------------------

/** A Triton or promoted model (the `detectors[]` and `ocr.*_models` entry). */
export interface VocabModel {
  name: string;
  choice: Choice;
  source: string;
  state?: string | null;
  ready?: boolean;
  versions?: string[];
  /** The deployment's configured model for this role (OCR lists). */
  configured?: boolean;
  /** PP §5.5: the owning project, whether it is shared, and how its
   *  classes map here. */
  project?: string | null;
  shared?: boolean;
  class_mapping?: { mapped: number; unmapped: string[] } | null;
  promoted_at?: string | null;
  job_id?: string | null;
}

export interface VocabSegmenter {
  name: string;
  choice: Choice;
  endpoint: string | null;
  status: string;
  masks: boolean;
  /** W8.4: the server's candidate cap and SAM 3's own score floor. */
  max_candidates: number | null;
  default_min_score: number | null;
}

export interface VocabVlmEndpoint {
  name: string;
  source: string;
  model: string | null;
  resolved_model: string | null;
  locality: string | null;
  sends_images_externally: boolean;
  status: string | null;
  max_images_per_call: number | null;
  active: boolean;
}

export interface VocabTextReaderMode {
  id: string;
  choice: Choice;
  label: string;
  reads_text: boolean;
  needs_vlm: boolean;
  needs_ocr: boolean;
}

export interface VocabRegistryClass {
  class_id: number;
  class_name: string;
  /** `choice.id` is the class NAME (`parent_classes` stores names). */
  choice: Choice;
}

export interface ConfigVocabulary {
  detectors: VocabModel[];
  segmenters: VocabSegmenter[];
  vlm: { active: ActiveRef | null; endpoints: VocabVlmEndpoint[] };
  ocr: {
    available: boolean;
    pipeline_models: VocabModel[];
    det_models: VocabModel[];
    rec_models: VocabModel[];
  };
  /** W9.8; rendered on the models page (Step 7), not here. */
  model_choices?: unknown[];
  text_reader_modes: VocabTextReaderMode[];
  registry_classes: VocabRegistryClass[];
  prompt_pack_calls?: Choice[];
  labels?: Record<string, Record<string, string>>;
}
