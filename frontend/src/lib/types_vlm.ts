/**
 * Wire types for OpenProcessor W9 (VLM endpoint registry, local model
 * catalog and per-project activation), field for field from the vendored
 * contract (`contracts/openprocessor/openapi/curation.json`, f582aa05).
 * The registry and catalog routes are GLOBAL (`{prefix}/vlm/...`, shared
 * by every project); activation is per project
 * (`{prefix}/projects/{project}/vlm/endpoints/...`).
 *
 * Body types are `type` aliases, not interfaces, so they satisfy the
 * shared editor's `Record<string, unknown>` body constraint.
 */
import type { ActiveConfigResponse, ActiveRef, ValidationReport } from './types_config';

/** The uniform `{id, label}` the VLM form's choice lists carry (`id` is
 *  `null` for an "empty" choice). */
export interface VlmChoice {
  id: string | null;
  label: string;
}

export type VlmLocality = 'compose' | 'host' | 'private' | 'external' | 'unknown';
export type VlmEndpointSource = 'env' | 'stored';
export type VlmEndpointStatus = 'ready' | 'unprobed' | 'probe_failed' | 'unreachable';
export type VlmJsonMode = 'auto' | 'on' | 'off';

/** `VlmEndpointBody` (strict). `api_key_ref` names a secret, never a key. */
export type VlmEndpointBody = {
  base_url: string;
  model: string;
  api_key_ref: string | null;
  catalog_id: string | null;
  allow_external: boolean;
  json_mode: VlmJsonMode;
  max_images_per_call: number;
  open_images_per_call: number | null;
  requests_per_second: number;
  timeout_s: number;
};

export interface SecretRef {
  ref: string;
  present: boolean;
  choice: VlmChoice;
}

export interface VlmEndpointLabels {
  status: Record<string, string>;
  locality: Record<string, string>;
  source: Record<string, string>;
}

/** One `GET /vlm/endpoints` row. */
export interface VlmEndpointSummary {
  name: string;
  source: VlmEndpointSource;
  read_only: boolean;
  revision: number | null;
  etag: string;
  description: string;
  base_url: string;
  model: string;
  catalog_id: string | null;
  locality: VlmLocality | null;
  sends_images_externally: boolean;
  warning: string | null;
  api_key_ref: string | null;
  api_key_present: boolean;
  status: VlmEndpointStatus;
  last_probe_at: string | null;
  active_in?: string[];
  updated_at?: string | null;
}

export interface VlmEndpointList {
  endpoints: VlmEndpointSummary[];
  config_revision: number;
  /** `ack`: an external endpoint needs an acknowledgement; `deny`: never. */
  external_policy: 'ack' | 'deny';
  secret_refs: SecretRef[];
  labels: VlmEndpointLabels;
  stale?: boolean;
}

export interface VlmEndpointFieldSchema {
  field: string;
  label: string;
  group: string;
  type: 'string' | 'int' | 'float' | 'bool' | 'enum';
  default: unknown;
  advanced?: boolean;
  choices_from?: 'secret_refs' | 'vlm_catalog' | null;
  empty_choice?: VlmChoice | null;
  enum?: VlmChoice[] | null;
  help?: string;
  min?: number | null;
  max?: number | null;
}

export interface VlmEndpointGroup {
  id: string;
  label: string;
}

export interface VlmEndpointSchema {
  fields: VlmEndpointFieldSchema[];
  groups: VlmEndpointGroup[];
}

/** `POST /vlm/endpoints/{name}/probe` and `validate?probe=true`. */
export interface VlmProbeResult {
  ok: boolean;
  probed_at: string;
  latency_ms?: number | null;
  models_listed?: string[];
  model_listed?: boolean | null;
  root?: string | null;
  max_model_len?: number | null;
  vision_ok?: boolean | null;
  json_mode_supported?: boolean | null;
  reasoning_channel?: boolean | null;
  image_tokens?: number | null;
  max_images_ok?: boolean | null;
  issues?: ValidationReport['errors'];
}

/** `GET /vlm/endpoints/{name}` and `.../revisions/{rev}`. Standalone (it
 *  satisfies the shared `ConfigDocBase`): a VLM doc serves `active_in`,
 *  never the base's `active` / `active_revision`, and its `validation`
 *  may be `null`. */
export interface VlmEndpointDoc {
  name: string;
  source: VlmEndpointSource;
  read_only: boolean;
  revision: number | null;
  etag: string;
  description: string;
  body: VlmEndpointBody;
  created_at: string | null;
  updated_at: string | null;
  updated_by: string | null;
  cloned_from: string | null;
  validation: ValidationReport | null;
  api_key_present: boolean;
  locality: VlmLocality | null;
  sends_images_externally: boolean;
  warning: string | null;
  active_in?: string[];
  last_probe?: VlmProbeResult | null;
}

export interface VlmEndpointCreate {
  name: string;
  description?: string;
  body: VlmEndpointBody;
}

export interface VlmEndpointSaveRequest {
  expected_revision: number;
  description?: string | null;
  body: VlmEndpointBody;
}

export interface VlmEndpointCloneRequest {
  new_name: string;
  revision?: number | null;
  description?: string | null;
}

export interface VlmRevisionSummary {
  revision: number;
  saved_at: string | null;
  cloned_from: string | null;
  description: string;
}

export interface VlmRevisionsResponse {
  name: string;
  revisions: VlmRevisionSummary[];
}

export interface VlmValidateRequest {
  name?: string | null;
  body: VlmEndpointBody;
}

export interface VlmValidateResponse {
  validation: ValidationReport;
  locality: VlmLocality | null;
  sends_images_externally: boolean;
  probe?: VlmProbeResult | null;
}

export interface VlmCatalogEntry {
  id: string;
  choice: VlmChoice;
  hf_repo: string;
  family: string;
  license: string;
  license_url: string;
  gated: boolean;
  params_b: number | null;
  quantization: string | null;
  context_max: number;
  max_model_len: number;
  max_images: number;
  vram_gb: number;
  disk_gb: number | null;
  status: 'tested' | 'to_verify';
  rank: number;
  multi_box_verified: boolean | null;
  text_reading_verified: boolean | null;
  fits: boolean | null;
  serving: boolean;
  desired: boolean;
}

export interface VlmCatalogLabels {
  status: Record<string, string>;
}

export interface VlmLocalServed {
  model: string | null;
  root: string | null;
  catalog_id: string | null;
  max_model_len: number | null;
}

export interface VlmLocalDesired {
  catalog_id: string;
  requested_at: string | null;
  command: string;
}

export interface VlmLocalStatus {
  configured: boolean;
  endpoint: string | null;
  served: VlmLocalServed | null;
  desired: VlmLocalDesired | null;
  restart_required: boolean;
  poll_after_s: number | null;
  gpu_total_gb: number | null;
  can_restart_from_api: boolean;
  reason: string;
}

export interface VlmCatalogResponse {
  entries: VlmCatalogEntry[];
  local: VlmLocalStatus;
  labels: VlmCatalogLabels;
}

export interface VlmLocalSelectRequest {
  catalog_id: string;
  force?: boolean;
}

/** `GET {scoped}/vlm/endpoints/active` and every activation write. */
export interface VlmActiveResponse extends ActiveConfigResponse {
  validation?: ValidationReport | null;
}

export interface VlmActivateRequest {
  revision: number | null;
  expected_active: ActiveRef | null;
  force: boolean;
  acknowledge_external?: boolean;
}

export interface VlmRollbackRequest {
  expected_active: ActiveRef | null;
}

export type VlmDeactivateRequest = VlmRollbackRequest;
