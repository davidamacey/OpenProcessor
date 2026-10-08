/**
 * W9 VLM fixtures, shaped after the contract's own models (f582aa05) in
 * neutral names: a stored local endpoint, a stored external one and the
 * read-only env built-in.
 */
import type { ValidationReport } from '$lib/types_config';
import type {
  VlmActiveResponse,
  VlmCatalogResponse,
  VlmEndpointBody,
  VlmEndpointDoc,
  VlmEndpointList,
  VlmEndpointSchema,
  VlmEndpointSummary,
  VlmLocalStatus,
  VlmProbeResult,
} from '$lib/types_vlm';

export const cleanReport = (): ValidationReport => ({
  ok: true,
  errors: [],
  warnings: [],
  force_allowed: false,
});

export const EXTERNAL_WARNING =
  'Crops are sent to a service outside this deployment (api.example.com).';

export function bodyFixture(over: Partial<VlmEndpointBody> = {}): VlmEndpointBody {
  return {
    base_url: 'http://vlm.internal:8000/v1',
    model: 'example/vision-7b',
    api_key_ref: null,
    catalog_id: null,
    allow_external: false,
    json_mode: 'auto',
    max_images_per_call: 8,
    open_images_per_call: 3,
    requests_per_second: 500,
    timeout_s: 240,
    ...over,
  };
}

export function summaryFixture(
  over: Partial<VlmEndpointSummary> = {},
): VlmEndpointSummary {
  return {
    name: 'local_vlm',
    source: 'stored',
    read_only: false,
    revision: 3,
    etag: 'vlm:local_vlm:3',
    description: 'The GPU box',
    base_url: 'http://vlm.internal:8000/v1',
    model: 'example/vision-7b',
    catalog_id: null,
    locality: 'private',
    sends_images_externally: false,
    warning: null,
    api_key_ref: null,
    api_key_present: false,
    status: 'ready',
    last_probe_at: '2026-10-01T09:00:00Z',
    active_in: ['alpha'],
    ...over,
  };
}

export function externalSummaryFixture(
  over: Partial<VlmEndpointSummary> = {},
): VlmEndpointSummary {
  return summaryFixture({
    name: 'cloud_vlm',
    revision: 1,
    etag: 'vlm:cloud_vlm:1',
    description: 'A hosted model',
    base_url: 'https://api.example.com/v1',
    model: 'vendor/vision-large',
    locality: 'external',
    sends_images_externally: true,
    warning: EXTERNAL_WARNING,
    api_key_ref: 'CLOUD_VLM_KEY',
    api_key_present: true,
    status: 'unprobed',
    last_probe_at: null,
    active_in: [],
    ...over,
  });
}

export function envSummaryFixture(
  over: Partial<VlmEndpointSummary> = {},
): VlmEndpointSummary {
  return summaryFixture({
    name: 'env_default',
    source: 'env',
    read_only: true,
    revision: null,
    etag: 'vlm:env_default',
    description: 'Set by the deployment',
    active_in: [],
    ...over,
  });
}

export function listFixture(over: Partial<VlmEndpointList> = {}): VlmEndpointList {
  return {
    endpoints: [summaryFixture(), externalSummaryFixture(), envSummaryFixture()],
    config_revision: 12,
    external_policy: 'ack',
    secret_refs: [
      {
        ref: 'CLOUD_VLM_KEY',
        present: true,
        choice: { id: 'CLOUD_VLM_KEY', label: 'CLOUD_VLM_KEY' },
      },
      {
        ref: 'MISSING_KEY',
        present: false,
        choice: { id: 'MISSING_KEY', label: 'MISSING_KEY' },
      },
    ],
    labels: {
      status: {
        ready: 'Ready',
        unprobed: 'Not probed yet',
        probe_failed: 'Probe failed',
        unreachable: 'Unreachable',
      },
      locality: {
        compose: 'Same stack',
        host: 'This host',
        private: 'Private network',
        external: 'Outside this deployment',
        unknown: 'Unknown',
      },
      source: { env: 'Set by the deployment', stored: 'Saved here' },
    },
    ...over,
  };
}

export function schemaFixture(): VlmEndpointSchema {
  return {
    groups: [
      { id: 'connection', label: 'Connection' },
      { id: 'limits', label: 'Limits' },
    ],
    fields: [
      {
        field: 'base_url',
        label: 'Base URL',
        group: 'connection',
        type: 'string',
        default: '',
        help: 'The OpenAI-compatible base URL.',
      },
      {
        field: 'model',
        label: 'Model',
        group: 'connection',
        type: 'string',
        default: '',
        help: '',
      },
      {
        field: 'api_key_ref',
        label: 'API key',
        group: 'connection',
        type: 'string',
        default: null,
        choices_from: 'secret_refs',
        empty_choice: { id: null, label: 'No key' },
        help: 'Names a secret on the host (cropwright-secret set NAME).',
      },
      {
        field: 'catalog_id',
        label: 'Catalog model',
        group: 'connection',
        type: 'string',
        default: null,
        choices_from: 'vlm_catalog',
        empty_choice: { id: null, label: 'Not a catalog model' },
        help: '',
        advanced: true,
      },
      {
        field: 'allow_external',
        label: 'Allow crops to leave the deployment',
        group: 'connection',
        type: 'bool',
        default: false,
        help: '',
      },
      {
        field: 'json_mode',
        label: 'JSON mode',
        group: 'limits',
        type: 'enum',
        default: 'auto',
        enum: [
          { id: 'auto', label: 'Automatic' },
          { id: 'on', label: 'Always on' },
          { id: 'off', label: 'Off' },
        ],
        help: '',
        advanced: true,
      },
      {
        field: 'max_images_per_call',
        label: 'Images per call',
        group: 'limits',
        type: 'int',
        default: 8,
        min: 1,
        max: 32,
        help: '',
      },
      {
        field: 'timeout_s',
        label: 'Timeout (s)',
        group: 'limits',
        type: 'float',
        default: 240,
        min: 1,
        help: '',
        advanced: true,
      },
    ],
  };
}

export function docFixture(over: Partial<VlmEndpointDoc> = {}): VlmEndpointDoc {
  return {
    name: 'local_vlm',
    source: 'stored',
    read_only: false,
    revision: 3,
    etag: 'vlm:local_vlm:3',
    description: 'The GPU box',
    body: bodyFixture(),
    created_at: '2026-09-30T10:00:00Z',
    updated_at: '2026-10-01T08:00:00Z',
    updated_by: null,
    cloned_from: null,
    validation: cleanReport(),
    api_key_present: false,
    locality: 'private',
    sends_images_externally: false,
    warning: null,
    active_in: ['alpha'],
    last_probe: null,
    ...over,
  };
}

export function revisionsFixture() {
  return {
    name: 'local_vlm',
    revisions: [
      {
        revision: 3,
        saved_at: '2026-10-01T08:00:00Z',
        cloned_from: null,
        description: 'The GPU box',
      },
      {
        revision: 2,
        saved_at: '2026-09-30T12:00:00Z',
        cloned_from: null,
        description: '',
      },
      { revision: 1, saved_at: null, cloned_from: null, description: '' },
    ],
  };
}

export function probeFixture(over: Partial<VlmProbeResult> = {}): VlmProbeResult {
  return {
    ok: true,
    probed_at: '2026-10-01T09:30:00Z',
    latency_ms: 84,
    models_listed: ['example/vision-7b'],
    model_listed: true,
    root: 'example/vision-7b',
    max_model_len: 32768,
    vision_ok: true,
    json_mode_supported: true,
    reasoning_channel: false,
    image_tokens: 256,
    max_images_ok: true,
    issues: [],
    ...over,
  };
}

export function activeFixture(over: Partial<VlmActiveResponse> = {}): VlmActiveResponse {
  return {
    axis: 'vlm',
    active: { name: 'local_vlm', revision: 3 },
    source: 'stored',
    activated_at: '2026-10-01T08:30:00Z',
    previous: { name: 'env_default', revision: null },
    config_revision: 12,
    stale: false,
    applied: [
      {
        process: 'vlm_worker',
        host: 'worker-1',
        applied_config_revision: 12,
        profile: { name: null, revision: null },
        pack: { name: null, revision: null },
        vlm: { name: 'local_vlm', revision: 3 },
        applied_at: '2026-10-01T08:30:02Z',
        lagging: false,
      },
    ],
    ...over,
  };
}

export function localFixture(over: Partial<VlmLocalStatus> = {}): VlmLocalStatus {
  return {
    configured: true,
    endpoint: 'local_vlm',
    served: {
      model: 'example/vision-7b',
      root: 'example/vision-7b',
      catalog_id: 'vision-7b',
      max_model_len: 32768,
    },
    desired: null,
    restart_required: false,
    poll_after_s: null,
    gpu_total_gb: 48,
    can_restart_from_api: false,
    reason: 'The local model server is serving the selected model.',
    ...over,
  };
}

export function catalogFixture(
  over: Partial<VlmCatalogResponse> = {},
): VlmCatalogResponse {
  return {
    entries: [
      {
        id: 'vision-7b',
        choice: { id: 'vision-7b', label: 'Vision 7B' },
        hf_repo: 'example/vision-7b',
        family: 'vision',
        license: 'Apache-2.0',
        license_url: 'https://example.com/license/apache',
        gated: false,
        params_b: 7,
        quantization: 'fp8',
        context_max: 32768,
        max_model_len: 16384,
        max_images: 8,
        vram_gb: 24,
        disk_gb: 16,
        status: 'tested',
        rank: 1,
        multi_box_verified: true,
        text_reading_verified: null,
        fits: true,
        serving: true,
        desired: false,
      },
      {
        id: 'vision-30b',
        choice: { id: 'vision-30b', label: 'Vision 30B' },
        hf_repo: 'example/vision-30b',
        family: 'vision',
        license: 'Custom',
        license_url: 'https://example.com/license/custom',
        gated: true,
        params_b: 30,
        quantization: null,
        context_max: 65536,
        max_model_len: 32768,
        max_images: 16,
        vram_gb: 80,
        disk_gb: null,
        status: 'to_verify',
        rank: 2,
        multi_box_verified: null,
        text_reading_verified: null,
        fits: false,
        serving: false,
        desired: false,
      },
    ],
    local: localFixture(),
    labels: { status: { tested: 'Tested', to_verify: 'To verify' } },
    ...over,
  };
}
