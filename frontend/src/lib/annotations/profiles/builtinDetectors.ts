/**
 * The built-in detector registry for this deployment's models — verbatim reproduction of
 * `DetectorChip.svelte`'s `labelFor`/`paletteFor`, so a future migration
 * to `ProvenanceChip.svelte` (§3.2 of the plan) renders pixel-identically.
 */

import type { DetectorRegistry } from '../detectorRegistry';

export const builtinDetectorRegistry: DetectorRegistry = {
  labels: {
    lpr_nanov11_640: 'LPR',
    sam3: 'SAM3',
    paddleocr_det_trt: 'Paddle det',
    paddleocr_rec_trt: 'Paddle rec',
    paddleocr_rec: 'Paddle rec',
    paddleocr_det: 'Paddle det',
    paddleocr: 'Paddle',
    human: 'Human',
    'gemma-4-e4b': 'Gemma',
    gemma: 'Gemma',
    gemma_propose: 'Gemma',
    gemma_prefilter: 'Gemma prefilter',
    vlm: 'VLM',
    yolov11_small_trt_end2end: 'YOLO11',
    coco_yolo11_proposal: 'YOLO11',
    onnxruntime: 'ORT',
    'ort-cuda': 'ORT·CUDA',
    'ort-trt': 'ORT·TRT',
    'ort-cpu': 'ORT·CPU',
    coreml: 'CoreML',
  },
  // Exact rules first — these are the three branches in DetectorChip's
  // if-chain that are NOT prefix tests and must not be shadowed by one.
  palettes: {
    human: 'emerald',
    coco_yolo11_proposal: 'sky',
    onnxruntime: 'indigo',
  },
  // Order reproduces DetectorChip.svelte's paletteFor top to bottom.
  prefixes: [
    { startsWith: 'lpr', palette: 'blue' },
    { startsWith: 'sam', palette: 'purple' },
    { startsWith: 'paddle', palette: 'amber' },
    { startsWith: 'gemma', palette: 'teal' },
    { startsWith: 'vlm', palette: 'teal' },
    { startsWith: 'yolov11', palette: 'sky' },
    { startsWith: 'ort', palette: 'indigo' },
    { startsWith: 'coreml', palette: 'orange' },
  ],
  fallback: 'zinc',
  // `accepted_unverified` (2026-09-24 logic-moves W8): a chain step the
  // backend accepted without a human/Gemma verification pass — muted,
  // same as a miss/reject, so it reads as "lower confidence" rather than
  // a confirmed step.
  mutedTagPattern: /miss|reject|skipped|degenerate|unparseable|accepted_unverified/,
};
