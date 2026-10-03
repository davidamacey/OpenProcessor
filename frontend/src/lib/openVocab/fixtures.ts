/**
 * Open-vocabulary fixtures in the neutral widget domain, shaped after the
 * contract's own models (fce17771). Shared by the unit and mount tests.
 */
import type {
  ActiveConfigResponse,
  ValidationIssue,
  ValidationReport,
} from '$lib/types_config';
import type {
  OpenVocabBody,
  OpenVocabDoc,
  OpenVocabFieldSchema,
  OpenVocabList,
  OpenVocabRevisionsResponse,
  OpenVocabSchema,
  OpenVocabSummary,
  OpenVocabTestResponse,
} from '$lib/types_openVocab';

export const cleanReport = (): ValidationReport => ({
  ok: true,
  errors: [],
  warnings: [],
  force_allowed: false,
});

export function issue(over: Partial<ValidationIssue> = {}): ValidationIssue {
  return {
    code: 'open_vocab_empty_prompt',
    severity: 'error',
    field: 'targets[0].prompt',
    message: 'A target needs a prompt.',
    detail: {},
    bypassable: false,
    ...over,
  };
}

export const errorReport = (...issues: ValidationIssue[]): ValidationReport => ({
  ok: false,
  errors: issues,
  warnings: [],
  force_allowed: false,
});

const row = (over: Partial<OpenVocabFieldSchema>): OpenVocabFieldSchema => ({
  scope: 'set',
  field: 'x',
  label: 'X',
  type: 'string',
  default: '',
  advanced: false,
  help: '',
  min: null,
  max: null,
  ...over,
});

export function schemaFixture(): OpenVocabSchema {
  return {
    max_enabled_targets_ceiling: 16,
    fields: [
      row({ scope: 'set', field: 'display_name', label: 'Display name' }),
      row({
        scope: 'set',
        field: 'run_on_ingest',
        label: 'Run on ingest',
        type: 'bool',
        default: false,
      }),
      row({
        scope: 'set',
        field: 'image_max_side',
        label: 'Longest image side',
        type: 'int',
        default: 1024,
        advanced: true,
        min: 256,
        max: 2048,
      }),
      row({ scope: 'target', field: 'prompt', label: 'Prompt', help: 'What to find.' }),
      row({ scope: 'target', field: 'class_name', label: 'Class name', default: '' }),
      row({
        scope: 'target',
        field: 'min_score',
        label: 'Minimum score',
        type: 'float',
        default: 0.5,
        min: 0,
        max: 1,
      }),
      row({
        scope: 'target',
        field: 'enabled',
        label: 'Enabled',
        type: 'bool',
        default: true,
      }),
      row({
        scope: 'target',
        field: 'max_instances',
        label: 'Most instances',
        type: 'int',
        default: 20,
        advanced: true,
      }),
      row({
        scope: 'target',
        field: 'parent_classes',
        label: 'Only inside these item classes',
        type: 'string_list',
        default: [],
        advanced: true,
      }),
      row({
        scope: 'gating',
        field: 'tier2_vlm_precheck',
        label: 'VLM pre-check',
        type: 'bool',
        default: false,
      }),
      row({
        scope: 'tier3_hit_rate',
        field: 'enabled',
        label: 'Hit-rate gate',
        type: 'bool',
        default: false,
        advanced: true,
      }),
      row({
        scope: 'tier3_hit_rate',
        field: 'window',
        label: 'Window',
        type: 'int',
        default: 20,
        advanced: true,
      }),
    ],
  };
}

export function bodyFixture(over: Partial<OpenVocabBody> = {}): OpenVocabBody {
  return {
    display_name: 'Widgets',
    run_on_ingest: false,
    image_max_side: 1024,
    dedup_iou: 0.5,
    max_enabled_targets: 8,
    targets: [
      {
        prompt: 'blue widget',
        class_name: 'widget',
        enabled: true,
        mask: true,
        min_score: 0.5,
        max_instances: 20,
        parent_classes: [],
      },
      { prompt: 'cracked widget', class_name: '', enabled: true, min_score: 0.4 },
    ],
    gating: { tier2_vlm_precheck: false, tier3_hit_rate: { enabled: false, window: 20 } },
    ...over,
  };
}

export function docFixture(over: Partial<OpenVocabDoc> = {}): OpenVocabDoc {
  return {
    name: 'widgets',
    source: 'stored',
    read_only: false,
    revision: 3,
    etag: 'ov:widgets:3',
    description: 'Widget finder',
    body: bodyFixture(),
    created_at: '2026-10-01T09:00:00Z',
    updated_at: '2026-10-02T09:00:00Z',
    updated_by: null,
    cloned_from: null,
    active: false,
    active_revision: null,
    validation: cleanReport(),
    ...over,
  };
}

export function summaryFixture(over: Partial<OpenVocabSummary> = {}): OpenVocabSummary {
  return {
    name: 'widgets',
    source: 'stored',
    read_only: false,
    revision: 3,
    etag: 'ov:widgets:3',
    display_name: 'Widgets',
    n_targets: 2,
    n_enabled_targets: 2,
    run_on_ingest: false,
    active: false,
    active_revision: null,
    updated_at: '2026-10-02T09:00:00Z',
    ...over,
  };
}

export function listFixture(over: Partial<OpenVocabList> = {}): OpenVocabList {
  return {
    sets: [summaryFixture()],
    templates: [
      {
        name: 'starter_widgets',
        path: 'templates/open_vocab/starter_widgets.json',
        n_targets: 1,
        display_name: 'Starter widgets',
        read_only: true,
        source: 'template',
      },
    ],
    active: { name: null, revision: null },
    config_revision: 7,
    stale: false,
    ...over,
  };
}

export function activeFixture(
  over: Partial<ActiveConfigResponse> = {},
): ActiveConfigResponse {
  return {
    axis: 'open_vocab',
    active: { name: null, revision: null },
    source: 'env',
    activated_at: null,
    previous: null,
    config_revision: 7,
    stale: false,
    applied: [],
    ...over,
  };
}

export function revisionsFixture(): OpenVocabRevisionsResponse {
  return {
    name: 'widgets',
    revisions: [
      {
        revision: 3,
        saved_at: '2026-10-02T09:00:00Z',
        cloned_from: null,
        description: '',
      },
      {
        revision: 2,
        saved_at: '2026-10-01T09:00:00Z',
        cloned_from: null,
        description: '',
      },
    ],
  };
}

export function testResponseFixture(
  over: Partial<OpenVocabTestResponse> = {},
): OpenVocabTestResponse {
  return {
    image: { width: 800, height: 600 },
    prompt: 'blue widget',
    class_name: 'widget',
    gate: { run: true, tier: null, reason: null },
    hits: [
      {
        bbox_norm: [0.1, 0.1, 0.4, 0.5],
        score: 0.91,
        selected: true,
        drop_reason: null,
        mask_polygon: [
          [0.1, 0.1],
          [0.4, 0.1],
          [0.4, 0.5],
        ],
      },
      {
        bbox_norm: [0.5, 0.5, 0.8, 0.9],
        score: 0.62,
        selected: false,
        drop_reason: 'agree_existing',
        mask_polygon: null,
      },
    ],
    elapsed_ms: 412.5,
    validation: cleanReport(),
    ...over,
  };
}
