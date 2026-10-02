/**
 * W4 region-profile fixtures, shaped after the spec's own examples
 * (any_domain_plan.md §7.3, §7.4) in the neutral widget/tag domain.
 */
import type { ActiveConfigResponse, ConfigRevisionList } from '$lib/types_config';
import type {
  ActivationImpact,
  ConfigVocabulary,
  ProfileActivateResponse,
  RegionProfileDoc,
  RegionProfileList,
  RegionProfileSchema,
} from '$lib/types_profiles';
import { cleanReport } from './promptPacks';

export const PROFILE_BODY = {
  display_name: 'Tags',
  display_name_singular: 'Tag',
  region_class_name: 'tag',
  parent_classes: ['widget'],
  detector_model: 'tag_detector_v1',
  segmenter_text_prompt: 'tag',
  max_regions_per_item: 4,
  segmenter_min_score: null,
  region_nms_iou: 0.5,
  text_reader: 'none',
  text_hint_enabled: false,
  input_size: 640,
  letterbox_fill: [114, 114, 114],
  auto_confirm_area_frac: [0.1, 0.9],
  ocr_det_model: '',
};

export function profileSchemaFixture(): RegionProfileSchema {
  return {
    groups: [
      { id: 'identity', label: 'Name and display' },
      { id: 'items', label: 'Which items' },
      { id: 'detector', label: 'Detector' },
      { id: 'segmenter', label: 'Segmenter' },
      { id: 'text', label: 'Text reading' },
      { id: 'advanced', label: 'Advanced' },
    ],
    fields: [
      {
        field: 'display_name',
        label: 'Display name (plural)',
        group: 'identity',
        type: 'string',
        default: '',
        min: null,
        max: null,
        enum: null,
        advanced: false,
        applies_when: null,
        choices_from: null,
        empty_choice: null,
        help: 'What the region is called in the app.',
      },
      {
        field: 'parent_classes',
        label: 'Item classes to search',
        group: 'items',
        type: 'string_list',
        default: [],
        min: null,
        max: null,
        enum: null,
        advanced: false,
        applies_when: null,
        choices_from: 'registry_classes',
        empty_choice: null,
        help: 'Only items of these classes get a region.',
      },
      {
        field: 'detector_model',
        label: 'Detector model',
        group: 'detector',
        type: 'string',
        default: '',
        min: null,
        max: null,
        enum: null,
        advanced: false,
        applies_when: null,
        choices_from: 'detectors',
        empty_choice: { id: '', label: 'No detector leg' },
        help: 'The model that proposes region boxes.',
      },
      {
        field: 'segmenter_text_prompt',
        label: 'Segmenter prompt',
        group: 'segmenter',
        type: 'string',
        default: '',
        min: null,
        max: 200,
        enum: null,
        advanced: false,
        applies_when: 'segmenter',
        choices_from: null,
        empty_choice: null,
        help: 'What the segmenter should find in each item crop.',
      },
      {
        field: 'max_regions_per_item',
        label: 'Most regions per item',
        group: 'segmenter',
        type: 'int',
        default: 1,
        min: 1,
        max: 64,
        enum: null,
        advanced: false,
        applies_when: null,
        choices_from: null,
        empty_choice: null,
        help: 'Machine proposals kept per item.',
      },
      {
        field: 'segmenter_min_score',
        label: 'Segmenter minimum score',
        group: 'segmenter',
        type: 'float',
        default: null,
        min: 0,
        max: 1,
        enum: null,
        advanced: false,
        applies_when: 'segmenter',
        choices_from: null,
        empty_choice: null,
        help: 'Candidates below this score are dropped.',
      },
      {
        field: 'region_nms_iou',
        label: 'Overlap for duplicates',
        group: 'segmenter',
        type: 'float',
        default: 0.5,
        min: 0,
        max: 1,
        enum: null,
        advanced: true,
        applies_when: null,
        choices_from: null,
        empty_choice: null,
        help: 'Boxes overlapping more than this are one region.',
      },
      {
        field: 'text_reader',
        label: 'Text reading',
        group: 'text',
        type: 'enum',
        default: 'vlm_then_ocr',
        min: null,
        max: null,
        enum: [
          { id: 'none', label: 'Off (region has no text)' },
          { id: 'vlm', label: 'VLM' },
          { id: 'ocr', label: 'OCR' },
        ],
        advanced: false,
        applies_when: null,
        choices_from: null,
        empty_choice: null,
        help: 'How region text is read.',
      },
      {
        field: 'text_hint_enabled',
        label: 'Use an OCR hint',
        group: 'text',
        type: 'bool',
        default: false,
        min: null,
        max: null,
        enum: null,
        advanced: false,
        applies_when: 'reads_text',
        choices_from: null,
        empty_choice: null,
        help: 'Give the VLM an OCR reading as a hint.',
      },
      {
        field: 'ocr_det_model',
        label: 'OCR detector',
        group: 'text',
        type: 'string',
        default: '',
        min: null,
        max: null,
        enum: null,
        advanced: true,
        applies_when: 'reads_text',
        choices_from: 'ocr_det_models',
        empty_choice: { id: '', label: 'None' },
        help: 'The OCR text detector.',
      },
      {
        field: 'input_size',
        label: 'Detector input size',
        group: 'advanced',
        type: 'int',
        default: 640,
        min: 32,
        max: 2048,
        enum: null,
        advanced: true,
        applies_when: 'detector',
        choices_from: null,
        empty_choice: null,
        help: 'A multiple of 32.',
      },
      {
        field: 'letterbox_fill',
        label: 'Letterbox fill',
        group: 'advanced',
        type: 'rgb',
        default: [114, 114, 114],
        min: 0,
        max: 255,
        enum: null,
        advanced: true,
        applies_when: 'detector',
        choices_from: null,
        empty_choice: null,
        help: 'Padding colour.',
      },
      {
        field: 'auto_confirm_area_frac',
        label: 'Auto-confirm area range',
        group: 'advanced',
        type: 'float_pair',
        default: [0.1, 0.9],
        min: 0,
        max: 1,
        enum: null,
        advanced: true,
        applies_when: null,
        choices_from: null,
        empty_choice: null,
        help: 'Box area fractions that confirm without review.',
      },
    ],
  };
}

export function profileDocFixture(
  over: Partial<RegionProfileDoc> = {},
): RegionProfileDoc {
  return {
    name: 'widget_tag',
    source: 'stored',
    read_only: false,
    revision: 3,
    etag: 'region_profile:widget_tag:3',
    description: 'Tags on widgets',
    body: structuredClone(PROFILE_BODY),
    effective: {
      reads_text: false,
      text_hint_active: false,
      legs: ['detector', 'segmenter'],
      segmenter_enabled: true,
    },
    created_at: '2026-09-26T10:00:00Z',
    updated_at: '2026-09-26T12:00:00Z',
    updated_by: null,
    cloned_from: 'template:widget_tag@-',
    active: true,
    active_revision: 2,
    validation: cleanReport(),
    ...over,
  };
}

export function profileListFixture(): RegionProfileList {
  return {
    profiles: [
      {
        name: 'env_tags',
        source: 'env',
        read_only: true,
        revision: null,
        etag: 'region_profile:env_tags:1a2b',
        display_name: 'Tags',
        display_name_singular: 'Tag',
        region_class_name: 'tag',
        text_reader: 'ocr',
        reads_text: true,
        detector_model: 'tag_detector_v1',
        segmenter_text_prompt: '',
        parent_classes: ['widget'],
        max_regions_per_item: 1,
        active: false,
        updated_at: null,
      },
      {
        name: 'widget_tag',
        source: 'stored',
        read_only: false,
        revision: 3,
        etag: 'region_profile:widget_tag:3',
        display_name: 'Tags',
        display_name_singular: 'Tag',
        region_class_name: 'tag',
        text_reader: 'none',
        reads_text: false,
        detector_model: '',
        segmenter_text_prompt: 'tag',
        parent_classes: ['widget', 'gadget'],
        max_regions_per_item: 4,
        active: true,
        active_revision: 2,
        updated_at: '2026-09-26T12:00:00Z',
      },
    ],
    templates: [
      {
        name: 'widget_tag',
        source: 'template',
        read_only: true,
        path: 'examples/region_profiles/widget_tag.json',
        display_name: 'Tags',
        reads_text: false,
      },
    ],
    active: { name: 'widget_tag', revision: 2 },
    config_revision: 21,
    stale: false,
  };
}

export function profileActiveFixture(
  over: Partial<ActiveConfigResponse> = {},
): ActiveConfigResponse {
  return {
    axis: 'detection_profile',
    active: { name: 'widget_tag', revision: 2 },
    source: 'stored',
    activated_at: '2026-09-26T12:05:00Z',
    previous: { name: 'env_tags', revision: null },
    config_revision: 21,
    stale: false,
    applied: [
      {
        process: 'detection_worker',
        host: 'worker-1',
        applied_config_revision: 21,
        profile: { name: 'widget_tag', revision: 2 },
        pack: { name: 'widget_tag', revision: 1 },
        applied_at: '2026-09-26T12:05:02Z',
        lagging: false,
      },
    ],
    ...over,
  };
}

export function impactFixture(over: Partial<ActivationImpact> = {}): ActivationImpact {
  return {
    items_total: 1840,
    by_profile: [
      { name: 'env_tags', revision: null, count: 900 },
      { name: 'widget_tag', revision: 3, count: 40 },
    ],
    validated_items: 12,
    unseeded_items: 850,
    pending_items: 38,
    pending_not_matching: 0,
    suggested_reprocess: {
      targets: {
        filter: { profile_not: 'widget_tag', include_detected: true },
      },
      scopes: ['region'],
      region_mode: 'redetect',
      dry_run: true,
    },
    ...over,
  };
}

export function activateResponseFixture(
  over: Partial<ProfileActivateResponse> = {},
): ProfileActivateResponse {
  return {
    ...profileActiveFixture({
      active: { name: 'widget_tag', revision: 3 },
      previous: { name: 'widget_tag', revision: 2 },
    }),
    impact: impactFixture(),
    validation: cleanReport(),
    ...over,
  };
}

export function profileRevisionsFixture(): ConfigRevisionList {
  return {
    name: 'widget_tag',
    revisions: [
      {
        revision: 3,
        saved_at: '2026-09-26T12:00:00Z',
        cloned_from: null,
        description: 'Tags on widgets',
      },
      {
        revision: 2,
        saved_at: '2026-09-26T11:00:00Z',
        cloned_from: null,
        description: 'Second cut',
      },
      {
        revision: 1,
        saved_at: '2026-09-26T10:00:00Z',
        cloned_from: 'template:widget_tag@-',
        description: 'First cut',
      },
    ],
  };
}

export function vocabularyFixture(): ConfigVocabulary {
  const model = (name: string, over: object = {}) => ({
    name,
    choice: { id: name, label: name },
    source: 'triton',
    state: 'READY',
    ready: true,
    versions: ['1'],
    ...over,
  });
  return {
    detectors: [
      model('tag_detector_v1', {
        choice: { id: 'tag_detector_v1', label: 'tag_detector_v1 (promoted)' },
        source: 'promoted',
        project: 'default',
        shared: false,
        class_mapping: { mapped: 1, unmapped: [] },
      }),
      model('item_detector_base'),
    ],
    segmenters: [
      {
        name: 'segmenter_v1',
        choice: { id: 'segmenter_v1', label: 'segmenter_v1' },
        endpoint: 'http://segmenter:8000',
        status: 'ready',
        masks: true,
        max_candidates: 128,
        default_min_score: 0.5,
      },
    ],
    vlm: {
      active: { name: 'env', revision: null },
      endpoints: [
        {
          name: 'env',
          source: 'env',
          model: 'local-vlm',
          resolved_model: 'example/vision-model',
          locality: 'compose',
          sends_images_externally: false,
          status: 'ready',
          max_images_per_call: 8,
          active: true,
        },
      ],
    },
    ocr: {
      available: true,
      pipeline_models: [model('ocr_pipeline', { configured: true })],
      det_models: [model('ocr_det_v1', { configured: true }), model('ocr_det_v2')],
      rec_models: [model('ocr_rec_v1', { configured: true })],
    },
    model_choices: [],
    text_reader_modes: [
      {
        id: 'none',
        choice: { id: 'none', label: 'Off (region has no text)' },
        label: 'Off (region has no text)',
        reads_text: false,
        needs_vlm: false,
        needs_ocr: false,
      },
      {
        id: 'ocr',
        choice: { id: 'ocr', label: 'OCR' },
        label: 'OCR',
        reads_text: true,
        needs_vlm: false,
        needs_ocr: true,
      },
    ],
    registry_classes: [
      { class_id: 0, class_name: 'widget', choice: { id: 'widget', label: 'widget' } },
      { class_id: 1, class_name: 'gadget', choice: { id: 'gadget', label: 'gadget' } },
      { class_id: 2, class_name: 'gizmo', choice: { id: 'gizmo', label: 'gizmo' } },
    ],
    prompt_pack_calls: [{ id: 'combined', label: 'Classify + verify region' }],
    labels: { scope: { per_run: 'Per run', region_profile: 'In the region profile' } },
  };
}
