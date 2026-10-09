/**
 * Served-shape combine fixtures (OpenProcessor P4, neutral widget/tag
 * domain). `combinePreview` mirrors `plan.py::build_preview`.
 */
import type {
  CombineJobResponse,
  CombinePreview,
  CombinePreviewSource,
} from '$lib/types_combine';

export function previewSource(
  project: string,
  classes: [string, number, string | null][],
): CombinePreviewSource {
  return {
    project,
    images: 10,
    items: classes.reduce((n, c) => n + c[1], 0),
    labeled_items: 4,
    holdout_images: 1,
    classes: classes.map(([name, count, mapped_to]) => ({ name, count, mapped_to })),
  };
}

export function combinePreview(over: Partial<CombinePreview> = {}): CombinePreview {
  return {
    ok: true,
    errors: [],
    warnings: [],
    preview_sha: 'sha-1',
    suggested_mapping: {
      'widgets-a': [
        { dataset_class: 'widget', action: 'create', new_class_name: 'widget' },
        { dataset_class: 'gadget', action: 'create', new_class_name: 'gadget' },
      ],
      'widgets-b': [{ dataset_class: 'widget', action: 'map', new_class_name: 'widget' }],
    },
    sources: [
      previewSource('widgets-a', [
        ['widget', 8, 'widget'],
        ['gadget', 2, 'gadget'],
      ]),
      previewSource('widgets-b', [['widget', 5, 'widget']]),
    ],
    target: {
      slug: 'merged',
      slug_available: true,
      classes: [
        {
          id: 0,
          name: 'widget',
          count: 13,
          from: [
            { project: 'widgets-a', class: 'widget' },
            { project: 'widgets-b', class: 'widget' },
          ],
        },
        {
          id: 1,
          name: 'gadget',
          count: 2,
          from: [{ project: 'widgets-a', class: 'gadget' }],
        },
      ],
      projected_images: 18,
      projected_items: 15,
      unclassed_items: 2,
      holdout_images: 1,
    },
    dedup: {
      identical_images: 2,
      merged_items: 1,
      conflicts: 1,
      conflict_samples: [
        {
          image_id: 'img-1',
          kept: { project: 'widgets-a', class: 'widget' },
          dropped: { project: 'widgets-b', class: 'gadget' },
        },
      ],
      near_duplicate_pairs_estimate: null,
    },
    bytes: { to_link: 2048, to_copy: 0 },
    ...over,
  };
}

export function combineJob(over: Partial<CombineJobResponse> = {}): CombineJobResponse {
  return {
    job_id: 'cmb_20261001T120000_1a2b3c4d',
    status: 'running',
    phase: 'images',
    done: 5,
    total: 20,
    started_at: '2026-10-01T12:00:00Z',
    finished_at: null,
    sources: ['widgets-a', 'widgets-b'],
    target: 'merged',
    error: null,
    report: {},
    next_steps: [],
    ...over,
  };
}
