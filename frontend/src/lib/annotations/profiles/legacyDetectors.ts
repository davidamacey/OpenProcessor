/**
 * The legacy deployment's detector registry — verbatim reproduction of
 * `DetectorChip.svelte`'s `labelFor`/`paletteFor`, so a future migration
 * to `ProvenanceChip.svelte` (§3.2 of the plan) renders pixel-identically.
 */

import type { DetectorRegistry } from '../detectorRegistry';

export const legacyDetectorRegistry: DetectorRegistry = {
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
    legacy_vehicle_v6_trt: 'v6',
    yolov11_small_trt_end2end: 'YOLO11',
    coco_yolo11_proposal: 'YOLO11',
    ingest_v6: 'Ingest v6',
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
    legacy_vehicle_v6_trt: 'rose',
    ingest_v6: 'rose',
    coco_yolo11_proposal: 'sky',
    onnxruntime: 'indigo',
  },
  // Order reproduces DetectorChip.svelte's paletteFor top to bottom.
  prefixes: [
    { startsWith: 'lpr', palette: 'blue' },
    { startsWith: 'sam', palette: 'purple' },
    { startsWith: 'paddle', palette: 'amber' },
    { startsWith: 'gemma', palette: 'teal' },
    { startsWith: 'yolov11', palette: 'sky' },
    { startsWith: 'ort', palette: 'indigo' },
    { startsWith: 'coreml', palette: 'orange' },
  ],
  fallback: 'zinc',
  mutedTagPattern: /miss|reject|skipped|degenerate|unparseable/,
};
