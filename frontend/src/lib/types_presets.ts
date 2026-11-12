/**
 * The research presets: serving profiles (deployment-wide), cluster profiles, assist-loop presets and
 * near-duplicate profiles (per project). Revisioned config documents whose body the server validates
 * on Save (`extra='forbid'`). Pinned to the OpenAPI by `contract/presetsContract.test.ts`.
 */
import type { ActiveRef } from '$lib/types_config';

export type PresetKindId =
  'serving_profile' | 'cluster_profile' | 'assist_loop_preset' | 'near_dup_profile';

export type PresetBody = Record<string, unknown>;

/** `ResearchConfigDoc`. */
export interface PresetDoc {
  kind: string;
  name: string;
  revision: number;
  description: string;
  body: PresetBody;
  created_at: string;
  updated_at: string;
  updated_by: string;
}

/** `ResearchConfigList`. */
export interface PresetListResponse {
  items: PresetDoc[];
}

/** One node of a JSON Schema (the parts pydantic writes that the help reads). */
export interface JsonSchemaNode {
  $ref?: string;
  $defs?: Record<string, JsonSchemaNode>;
  type?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  const?: unknown;
  anyOf?: JsonSchemaNode[];
  properties?: Record<string, JsonSchemaNode>;
  required?: string[];
  items?: JsonSchemaNode;
  additionalProperties?: boolean | JsonSchemaNode;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  minLength?: number;
  minItems?: number;
}

/** `ResearchAxis`: a place a project kind can be active. `target` is the body `target` the axis serves. */
export interface PresetAxisWire {
  axis: string;
  label: string;
  target: string | null;
}

/** `ResearchConfigSchema`: the JSON Schema of a kind's body and the axes it activates on, as served. */
export interface PresetSchemaWire {
  kind: string;
  axes: PresetAxisWire[];
  json_schema: JsonSchemaNode;
}

/** `ResearchConfigPut`: `expected_revision: null` creates the document. */
export interface PresetPutRequest {
  body: PresetBody;
  description: string;
  expected_revision: number | null;
}

/** `ResearchActivateRequest` (a project kind). */
export interface PresetActivateRequest {
  axis: string | null;
  revision: number | null;
  expected_active: ActiveRef | null;
}

/** `ServingActivateRequest`: `expected_revision` is the `setting_revision` last read. */
export interface ServingActivateRequest {
  revision: number | null;
  expected_revision: number | null;
}

/** `ResearchActive`: what one axis of a project kind has active. */
export interface PresetActiveWire {
  axis: string;
  state: 'none' | 'active' | 'off';
  name?: string | null;
  revision?: number | null;
  activated_at?: string | null;
}

/** `ServingActive`: the deployment's active serving profile, with the revision of the setting. */
export interface ServingActiveWire {
  name: string;
  revision: number;
  setting_revision: number;
  body: PresetBody;
}

/** `ServingActiveState`: `GET /serving_profiles/active`; `active` is null until one is activated. */
export interface ServingActiveStateWire {
  active: ServingActiveWire | null;
}

/** What is active on one axis, whichever kind served it. */
export interface ActivePreset {
  /** The project axis; null for the deployment's serving profile. */
  axis: string | null;
  state: 'none' | 'active' | 'off';
  name: string | null;
  revision: number | null;
  activated_at: string | null;
  /** Serving only: the revision of the active-profile setting, the compare-and-set token. */
  setting_revision: number | null;
}
