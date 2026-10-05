/**
 * W3 prompt-pack fixtures, shaped after the spec's own examples
 * (any_domain_plan.md §7.1, §7.2, §7.5) in the neutral widget/tag domain.
 */
import type {
  ActiveConfigResponse,
  ConfigRevisionList,
  ValidationIssue,
  ValidationReport,
} from '$lib/types_config';
import type { PromptPackDoc, PromptPackList, PromptPackSchema } from '$lib/types_packs';

export const cleanReport = (): ValidationReport => ({
  ok: true,
  errors: [],
  warnings: [],
  force_allowed: false,
});

export function issue(over: Partial<ValidationIssue> = {}): ValidationIssue {
  return {
    code: 'pack_placeholder_missing',
    id: 'pack_placeholder_missing:class_user_template',
    severity: 'error',
    field: 'class_user_template',
    message: 'class_user_template must contain {class_names_csv}',
    detail: { placeholder: 'class_names_csv' },
    bypassable: false,
    ...over,
  };
}

export function schemaFixture(): PromptPackSchema {
  return {
    fields: [
      {
        field: 'class_system',
        label: 'Classify: system message',
        group: 'classify',
        kind: 'text',
        formatted: false,
        required_placeholders: [],
        allowed_placeholders: [],
        expected_reply_keys: [],
        optional_reply_keys: [],
        used_by: ['auto_label_vlm_stage'],
        help: 'Sent verbatim as the system message.',
      },
      {
        field: 'class_user_template',
        label: 'Classify: user message',
        group: 'classify',
        kind: 'text',
        formatted: true,
        required_placeholders: ['class_names_csv'],
        allowed_placeholders: ['class_names_csv'],
        expected_reply_keys: ['img', 'class', 'confidence'],
        optional_reply_keys: [],
        used_by: ['auto_label_vlm_stage', 'vlm_label_batch'],
        help: 'Asks for one class per image.',
      },
      {
        field: 'combined_user_template',
        label: 'Classify + verify: user message',
        group: 'combined',
        kind: 'text',
        formatted: true,
        required_placeholders: ['class_block', 'region_block'],
        allowed_placeholders: ['class_block', 'region_block'],
        expected_reply_keys: ['class_id', 'class_confidence', 'region_boxes', 'box'],
        optional_reply_keys: ['region_set_complete'],
        used_by: ['detection_worker_combined_verify'],
        help: 'Classifies the item and judges each numbered box.',
      },
      {
        field: 'synonyms',
        label: 'Synonyms',
        group: 'vocabulary',
        kind: 'map',
        formatted: false,
        required_placeholders: [],
        allowed_placeholders: [],
        expected_reply_keys: [],
        optional_reply_keys: [],
        used_by: ['auto_label_vlm_stage'],
        help: 'Maps a word the VLM may answer to a registry class.',
      },
      {
        // Served as `kind: 'list'` since OpenProcessor 05ec48a8 (a list of
        // case-insensitive globs).
        field: 'proposal_denylist',
        label: 'Proposal denylist',
        group: 'vocabulary',
        kind: 'list',
        formatted: false,
        required_placeholders: [],
        allowed_placeholders: [],
        expected_reply_keys: [],
        optional_reply_keys: [],
        used_by: ['auto_label_vlm_stage'],
        help: '',
      },
    ],
    placeholders: [
      {
        name: 'class_names_csv',
        meaning: 'comma-separated class names from the registry',
        example: 'widget, gadget',
      },
      {
        name: 'class_block',
        meaning: "the classify instruction and numbered catalog, or 'don't classify'",
        example: '1. widget\n2. gadget',
      },
    ],
    calls: [
      {
        id: 'classify',
        label: 'Classify',
        fields: ['class_system', 'class_user_template'],
        testable: true,
      },
      {
        id: 'combined',
        label: 'Classify + verify region',
        fields: ['combined_user_template'],
        testable: true,
      },
    ],
    reply_key_contract: {
      classify: { required: ['img', 'class', 'confidence'], optional: [] },
    },
  };
}

export function docFixture(over: Partial<PromptPackDoc> = {}): PromptPackDoc {
  return {
    name: 'widget_tag',
    source: 'stored',
    read_only: false,
    revision: 2,
    etag: 'prompt_pack:widget_tag:2',
    description: 'Tags on widgets',
    body: {
      class_system: 'You classify widgets.',
      class_user_template: 'Pick one of: {class_names_csv}',
      combined_user_template: '{class_block}{region_block}',
      synonyms: { doohickey: 'gadget' },
      proposal_denylist: ['blurry_*', '*_scene'],
    },
    created_at: '2026-09-26T10:00:00Z',
    updated_at: '2026-09-26T12:00:00Z',
    updated_by: null,
    cloned_from: 'template:widget_tag',
    active: true,
    active_revision: 1,
    validation: cleanReport(),
    ...over,
  };
}

export function builtinDocFixture(): PromptPackDoc {
  return docFixture({
    name: 'generic_item_v1',
    source: 'builtin',
    read_only: true,
    revision: null,
    etag: 'prompt_pack:generic_item_v1:3f2a9c01be77',
    description: 'Built-in example',
    cloned_from: null,
    active: false,
    active_revision: null,
    created_at: null,
    updated_at: null,
  });
}

export function listFixture(): PromptPackList {
  return {
    packs: [
      {
        name: 'generic_item_v1',
        source: 'builtin',
        read_only: true,
        revision: null,
        etag: 'prompt_pack:generic_item_v1:3f2a9c01be77',
        description: 'Built-in example',
        asks_region_text: true,
        active: false,
        updated_at: null,
      },
      {
        name: 'widget_tag',
        source: 'stored',
        read_only: false,
        revision: 2,
        etag: 'prompt_pack:widget_tag:2',
        description: 'Tags on widgets',
        asks_region_text: false,
        active: true,
        active_revision: 1,
        updated_at: '2026-09-26T12:00:00Z',
      },
    ],
    templates: [
      {
        name: 'widget_tag',
        source: 'template',
        read_only: true,
        path: 'examples/prompt_packs/widget_tag.json',
      },
    ],
    active: { name: 'widget_tag', revision: 1 },
    config_revision: 17,
    stale: false,
  };
}

export function activeFixture(
  over: Partial<ActiveConfigResponse> = {},
): ActiveConfigResponse {
  return {
    axis: 'prompt_pack',
    active: { name: 'widget_tag', revision: 1 },
    source: 'stored',
    activated_at: '2026-09-26T12:05:00Z',
    previous: { name: 'generic_item_v1', revision: null },
    config_revision: 17,
    stale: false,
    applied: [
      {
        process: 'detection_worker',
        host: 'worker-1',
        applied_config_revision: 17,
        profile: { name: 'widget_tag', revision: 3 },
        pack: { name: 'widget_tag', revision: 1 },
        vlm: null,
        applied_at: '2026-09-26T12:05:02Z',
        lagging: false,
      },
    ],
    ...over,
  };
}

export function revisionsFixture(): ConfigRevisionList {
  return {
    name: 'widget_tag',
    revisions: [
      {
        revision: 2,
        saved_at: '2026-09-26T12:00:00Z',
        cloned_from: null,
        description: 'Tags on widgets',
      },
      {
        revision: 1,
        saved_at: '2026-09-26T10:00:00Z',
        cloned_from: 'template:widget_tag',
        description: 'First cut',
      },
    ],
  };
}
