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
      { format: 'auto', label: 'Detect automatically' },
      { format: 'yolo', label: 'YOLO (data.yaml)' },
      { format: 'coco', label: 'COCO JSON' },
      { format: 'openprocessor_export', label: 'OpenProcessor export' },
    ],
    processing_modes: [
      {
        value: 'none',
        label: 'Import as-is',
        description: 'Index the images and import the labels. No detector or VLM runs.',
      },
      {
        value: 'propose',
        label: 'Import and find missed objects',
        description: 'Also run the detector, region and VLM pipeline.',
      },
    ],
    parents_modes: [
      { value: 'auto', label: 'Automatic', description: '' },
      { value: 'labels', label: "From the dataset's labels", description: '' },
      { value: 'detect', label: 'Detect them', description: '' },
    ],
    trust_levels: [
      { value: 'validated', label: 'Trusted (validated)', description: '' },
      { value: 'suggestion', label: 'Suggestions to review', description: '' },
    ],
    mapping_actions: [
      { value: 'map', label: 'Map to class', description: '' },
      { value: 'create', label: 'Create class', description: '' },
      { value: 'skip', label: 'Skip', description: '' },
      { value: 'region', label: 'Region boxes', description: '' },
    ],
    match_kinds: [
      { value: 'exact', label: 'Same name', description: '' },
      { value: 'case_insensitive', label: 'Same name, different case', description: '' },
      { value: 'synonym', label: 'Synonym', description: '' },
      { value: 'none', label: 'No match', description: '' },
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
    upload_limits: {
      max_bytes: 2147483648,
      max_files: 200000,
      ttl_hours: 24,
      preview_max_files: 5000,
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
      images_failed: 0,
      images_skipped: 0,
      items_reconciled_removed: 0,
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
      {
        dataset_class: 'Widget',
        kind: 'item',
        class_id: 2,
        class_name: 'widget',
        created: false,
      },
    ],
    options: {},
    source: {
      format: 'yolo',
      root: '/data/source/widgets/yolo',
      source_sha: 'a',
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
    items_kept_shared: 0,
    items_reinstated: 0,
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
      {
        scope: 'region',
        selected: 12,
        locked_skipped: 3,
        queued: 0,
        failed: 0,
        not_found: 0,
        breakdown: [],
      },
    ],
    job: null,
    items: [],
    ...over,
  };
}
