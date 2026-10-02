/**
 * W10 wire fixtures, lifted from the spec's own examples
 * (any_domain_plan.md §7.12 JSON, W10.4, W10.12, W10.13) with the
 * neutral fixture domain's class names (widgets, gadgets).
 */
import type {
  DatasetFormatsResponse,
  DatasetImportJob,
  DatasetPreview,
  DatasetUndoReport,
  ReprocessResponse,
} from '$lib/types_import';

export function formatsFixture(
  over: Partial<DatasetFormatsResponse> = {},
): DatasetFormatsResponse {
  return {
    formats: [
      { id: 'auto', label: 'Detect automatically' },
      { id: 'yolo', label: 'YOLO (data.yaml)' },
      { id: 'coco', label: 'COCO JSON' },
      { id: 'openprocessor_export', label: 'OpenProcessor export' },
    ],
    processing: [
      {
        id: 'none',
        label: 'Import as-is',
        description: 'Index the images and import the labels. No detector or VLM runs.',
      },
      {
        id: 'propose',
        label: 'Import and find missed objects',
        description: 'Also run the detector, region and VLM pipeline.',
      },
    ],
    parents: [
      { id: 'auto', label: 'Automatic' },
      { id: 'labels', label: "From the dataset's labels" },
      { id: 'detect', label: 'Detect them' },
    ],
    label_trust: [
      { id: 'validated', label: 'Trusted (validated)' },
      { id: 'suggestion', label: 'Suggestions to review' },
    ],
    mapping_actions: [
      { id: 'map', label: 'Map to class' },
      { id: 'create', label: 'Create class' },
      { id: 'skip', label: 'Skip' },
      { id: 'region', label: 'Region boxes' },
    ],
    match_kinds: [
      { id: 'exact', label: 'Same name' },
      { id: 'case_insensitive', label: 'Same name, different case' },
      { id: 'synonym', label: 'Synonym' },
      { id: 'none', label: 'No match' },
    ],
    issues: [
      {
        code: 'label_file_missing',
        severity: 'warning',
        blocking: false,
        bypassable: false,
        label: 'Images without a label file',
      },
      {
        code: 'test_split_changed',
        severity: 'error',
        blocking: true,
        bypassable: true,
        label: 'The frozen test split changed',
      },
    ],
    upload: {
      max_bytes: 2147483648,
      max_files: 200000,
      accepted: ['.zip', '.tar', '.tar.gz'],
    },
    status_labels: {
      queued: 'Queued',
      running: 'Importing',
      paused_backpressure: 'Waiting for the region worker',
      completed: 'Done',
      completed_with_errors: 'Done with errors',
      failed: 'Failed',
      cancelled: 'Cancelled',
      interrupted: 'Interrupted',
      undoing: 'Undoing',
      undone: 'Undone',
    },
    reprocess: {
      scopes: [
        {
          id: 'detect',
          label: 'Detect again',
          description: 'Re-run the detectors.',
          unit: 'image',
        },
        {
          id: 'region',
          label: 'Regions',
          description: 'Regenerate machine regions.',
          unit: 'item',
        },
        {
          id: 'vlm',
          label: 'VLM class',
          description: 'Clear the VLM class.',
          unit: 'item',
        },
        {
          id: 'embed',
          label: 'Embeddings',
          description: 'Recompute vectors.',
          unit: 'item',
        },
      ],
      region_modes: [
        { id: 'redetect', label: 'Detect again', description: 'Remove machine boxes.' },
        {
          id: 'reverify',
          label: 'Verify again',
          description: 'Re-verify machine boxes.',
        },
      ],
      lock_rule:
        'Human and imported labels are never changed. Reprocess only regenerates machine proposals.',
    },
    ...over,
  };
}

export function previewFixture(over: Partial<DatasetPreview> = {}): DatasetPreview {
  return {
    project: 'default',
    format: 'yolo',
    root: '/data/source/widgets/yolo',
    source_sha: 'a'.repeat(64),
    import_key: 'k'.repeat(64),
    op_export: null,
    splits: [
      { split: 'train', images: 67, labeled: 58, negatives: 8, unlabeled: 1, boxes: 212 },
      { split: 'test', images: 15, labeled: 13, negatives: 2, unlabeled: 0, boxes: 44 },
    ],
    totals: { images: 82, boxes: 256, images_already_indexed: 0, images_to_ingest: 82 },
    classes: [
      {
        dataset_class: 'Widget',
        dataset_id: 0,
        boxes: 150,
        images: 70,
        source_class_id: null,
        index_would_have_mapped_to: { class_id: 1, class_name: 'gadget' },
        suggestion: {
          action: 'map',
          class_id: 2,
          class_name: 'widget',
          match: 'case_insensitive',
        },
        resolved: null,
      },
      {
        dataset_class: 'sprocket',
        dataset_id: 1,
        boxes: 31,
        images: 20,
        source_class_id: null,
        index_would_have_mapped_to: null,
        suggestion: {
          action: 'create',
          class_id: null,
          class_name: 'sprocket',
          match: 'none',
        },
        resolved: null,
      },
    ],
    region: null,
    issues: [
      {
        code: 'label_file_missing',
        severity: 'warning',
        blocking: false,
        bypassable: false,
        message: '1 image has no label file.',
        count: 1,
        samples: [{ file: 'images/train/w_0007.jpg', line: null, detail: {} }],
      },
    ],
    blocking: false,
    force_allowed: false,
    estimate: { detector_images: 0, embeddings: 181 },
    ...over,
  };
}

export function jobFixture(over: Partial<DatasetImportJob> = {}): DatasetImportJob {
  return {
    project: 'default',
    import_id: 'imp_20260927T120000_1a2b3c4d',
    import_key: 'k'.repeat(64),
    name: 'widgets_v1',
    status: 'running',
    reused: false,
    progress: {
      images_total: 82,
      images_done: 40,
      images_failed: 0,
      chunks_total: 2,
      chunks_done: 1,
      images_per_s: 21.5,
      eta_s: 2,
    },
    waiting_for: null,
    report: {
      images_created: 40,
      images_reused: 0,
      items_created: 120,
      items_updated: 0,
      items_noop: 0,
      labels_written: 120,
      boxes_written: 0,
      standalone_regions: 0,
      negatives: 4,
      unlabeled: 1,
      parents_detected: 0,
      proposals_created: 0,
      proposals_merged: 0,
      holdout_frozen: 0,
      label_conflicts_locked: 0,
      disagreements: { counts: {}, samples: [] },
    },
    mapping: [
      { dataset_class: 'Widget', kind: 'item', class_id: 2, class_name: 'widget' },
    ],
    options: {},
    source: {
      format: 'yolo',
      root: '/data/source/widgets/yolo',
      source_sha: 'a',
      op_export: null,
    },
    issues_summary: [],
    undo: null,
    next_steps: [],
    started_at: '2026-09-27T12:00:00Z',
    updated_at: '2026-09-27T12:00:02Z',
    finished_at: null,
    poll_after_s: 2,
    labels: {
      status: {
        running: 'Importing',
        completed: 'Done',
        failed: 'Failed',
        cancelled: 'Cancelled',
        interrupted: 'Interrupted',
        undoing: 'Undoing',
        undone: 'Undone',
      },
    },
    error: null,
    ...over,
  };
}

export function undoReportFixture(
  over: Partial<DatasetUndoReport> = {},
): DatasetUndoReport {
  return {
    import_id: 'imp_20260927T120000_1a2b3c4d',
    dry_run: true,
    items_deleted: 240,
    items_restored: 3,
    items_kept_human_edited: 2,
    class_labels_removed: 243,
    boxes_removed: 0,
    boxes_kept_human_edited: 0,
    proposals_deleted: 0,
    holdout_flags_cleared: 41,
    images_deleted: 78,
    images_kept: 4,
    classes_deprecated: ['sprocket'],
    ...over,
  };
}

export function reprocessFixture(
  over: Partial<ReprocessResponse> = {},
): ReprocessResponse {
  return {
    dry_run: true,
    scopes: [
      { scope: 'region', selected: 12, locked_skipped: 3, queued: 0, breakdown: [] },
    ],
    job: null,
    items: [],
    message: '12 items selected; 3 are locked and will be skipped.',
    ...over,
  };
}
