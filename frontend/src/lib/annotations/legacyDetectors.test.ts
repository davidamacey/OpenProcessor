import { describe, it, expect } from 'vitest';
import { labelForDetector, paletteForDetector, PALETTES } from './detectorRegistry';
import { legacyDetectorRegistry } from './profiles/legacyDetectors';

/**
 * Equivalence proof for the legacy detector registry (P2.2 in the
 * plan): every id `DetectorChip.svelte`'s `labelFor`/`paletteFor`
 * switch/if-chain handles today, plus one unknown id and null, snapshot
 * against a frozen `{label, palette}` pair captured directly from those
 * functions (see docs/genericization-plan-2026-09-13.md §3.2). This is
 * the proof that a future migration to a config-driven `ProvenanceChip`
 * renders pixel-identically — nothing consumes this registry in a
 * component yet.
 */
const KNOWN_IDS: Array<{ id: string | null; label: string; palette: string }> = [
  { id: 'lpr_nanov11_640', label: 'LPR', palette: 'blue' },
  { id: 'sam3', label: 'SAM3', palette: 'purple' },
  { id: 'paddleocr_det_trt', label: 'Paddle det', palette: 'amber' },
  { id: 'paddleocr_rec_trt', label: 'Paddle rec', palette: 'amber' },
  { id: 'paddleocr_rec', label: 'Paddle rec', palette: 'amber' },
  { id: 'paddleocr_det', label: 'Paddle det', palette: 'amber' },
  { id: 'paddleocr', label: 'Paddle', palette: 'amber' },
  { id: 'human', label: 'Human', palette: 'emerald' },
  { id: 'gemma-4-e4b', label: 'Gemma', palette: 'teal' },
  { id: 'gemma', label: 'Gemma', palette: 'teal' },
  { id: 'gemma_propose', label: 'Gemma', palette: 'teal' },
  { id: 'gemma_prefilter', label: 'Gemma prefilter', palette: 'teal' },
  { id: 'legacy_vehicle_v6_trt', label: 'v6', palette: 'rose' },
  { id: 'yolov11_small_trt_end2end', label: 'YOLO11', palette: 'sky' },
  { id: 'coco_yolo11_proposal', label: 'YOLO11', palette: 'sky' },
  { id: 'ingest_v6', label: 'Ingest v6', palette: 'rose' },
  { id: 'onnxruntime', label: 'ORT', palette: 'indigo' },
  { id: 'ort-cuda', label: 'ORT·CUDA', palette: 'indigo' },
  { id: 'ort-trt', label: 'ORT·TRT', palette: 'indigo' },
  { id: 'ort-cpu', label: 'ORT·CPU', palette: 'indigo' },
  { id: 'coreml', label: 'CoreML', palette: 'orange' },
  {
    id: 'some_unknown_future_detector',
    label: 'some_unknown_future_detector',
    palette: 'zinc',
  },
  { id: null, label: '—', palette: 'zinc' },
];

describe('legacyDetectorRegistry equivalence with DetectorChip.svelte', () => {
  it.each(KNOWN_IDS)(
    'resolves $id to {label: $label, palette: $palette}',
    ({ id, label, palette }) => {
      expect(labelForDetector(legacyDetectorRegistry, id)).toBe(label);
      expect(paletteForDetector(legacyDetectorRegistry, id)).toEqual(
        PALETTES[palette as keyof typeof PALETTES],
      );
    },
  );

  it('checks exact palette rules before prefix rules (order matters)', () => {
    // 'onnxruntime' is both an exact rule (indigo) and would match no
    // prefix; 'ort-cuda' is NOT an exact rule and must fall through to
    // the 'ort' prefix rule, also indigo — proving both paths agree.
    expect(paletteForDetector(legacyDetectorRegistry, 'onnxruntime')).toEqual(
      PALETTES.indigo,
    );
    expect(paletteForDetector(legacyDetectorRegistry, 'ort-cuda')).toEqual(
      PALETTES.indigo,
    );
  });

  it('mutes miss/reject/skipped/degenerate/unparseable tags', () => {
    for (const tag of [
      'miss',
      'reject',
      'skipped',
      'degenerate',
      'unparseable',
      'gemma_reject',
    ]) {
      expect(legacyDetectorRegistry.mutedTagPattern.test(tag)).toBe(true);
    }
    expect(legacyDetectorRegistry.mutedTagPattern.test('hit')).toBe(false);
  });
});

// B3 renamed the VLM's chain entries (`vlm_visible:yes`,
// `<det>:vlm_verify_ok`'s head is the detector) and its class_source to `vlm`.
describe('B3 VLM vocabulary', () => {
  it('colors vlm chain entries like the VLM model itself', () => {
    expect(paletteForDetector(legacyDetectorRegistry, 'vlm_visible:yes')).toBe(
      paletteForDetector(legacyDetectorRegistry, 'gemma-4-e4b'),
    );
    expect(paletteForDetector(legacyDetectorRegistry, 'vlm')).toBe(PALETTES.teal);
  });

  it('labels the vlm writer', () => {
    expect(labelForDetector(legacyDetectorRegistry, 'vlm')).toBe('VLM');
  });
});
