/**
 * `<CropMetaPanel>`'s per-box region section (mount-based): every box of
 * the served `region_boxes` list is its own block, numbered by position,
 * with that box's own state, score, text reading, candidate readings,
 * verdict and rejection. The item-level section keeps the human-validated
 * vs auto-confirmed split and the incomplete-set flag.
 */
import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import CropMetaPanel from './CropMetaPanel.svelte';
import { mapCropSlots } from '$lib/annotations/cropSlots';
import {
  installServedRegionProfile,
  resetDeploymentSlots,
} from '$lib/annotations/registeredSlots';
import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';
import { WIDGET_TAG_PROFILE } from '$lib/test/fixtures/regionSlot';
import type { Crop } from '$lib/types';

let target: HTMLDivElement;
let instance: unknown;

beforeAll(() => installServedRegionProfile(WIDGET_TAG_PROFILE));
afterAll(() => resetDeploymentSlots());

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network in test')));
});

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  regionVocabularyStore.resetForProjectChange();
  vi.unstubAllGlobals();
});

const wireBox = (over: Record<string, unknown>) => ({
  box_id: 'b1',
  state: 'accepted',
  bbox_norm: [0.4, 0.4, 0.6, 0.6],
  bbox_in_parent: [0.4, 0.4, 0.6, 0.6],
  score: 0.9,
  ...over,
});

function render(raw: Record<string, unknown>): HTMLDivElement {
  const crop = {
    id: 'crop-1',
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.5, h: 0.5 },
    class_id: 3,
    class_name: 'widget_a',
    cluster_id: 7,
    label_confidence: 0.9,
    label_source: 'model',
    class_source: 'model',
    updated_at: '',
    slots: mapCropSlots(raw, [0, 0, 1, 1]),
  } as unknown as Crop;
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CropMetaPanel, { target, props: { crop } } as never);
  flushSync();
  return target;
}

const blocks = (el: HTMLElement) =>
  [...el.querySelectorAll('[data-testid="meta-box"]')].map((b) =>
    (b.textContent ?? '').replace(/\s+/g, ' ').trim(),
  );

describe('per-box blocks', () => {
  it('renders one numbered block per served box, each with its own score and text', () => {
    const el = render({
      region_status: 'detected',
      region_boxes: [
        wireBox({ box_id: 'b1', score: 0.9, text: 'TAG-001', text_confidence: 0.8 }),
        wireBox({ box_id: 'b2', state: 'rejected', score: 0.42, text: 'TAG-002' }),
      ],
    });
    const b = blocks(el);
    expect(b).toHaveLength(2);
    expect(b[0]).toContain('#1');
    expect(b[0]).toContain('TAG-001');
    expect(b[0]).toContain('90.0%');
    expect(b[1]).toContain('#2');
    expect(b[1]).toContain('TAG-002');
    expect(b[1]).toContain('42.0%');
  });

  it('shows a "model: box wrong" chip only on the box the verifier judged wrong', () => {
    const el = render({
      region_boxes: [
        wireBox({ box_id: 'b1', bbox_correct: true }),
        wireBox({ box_id: 'b2', bbox_correct: false }),
      ],
    });
    const b = blocks(el);
    expect(b[0]).not.toContain('model: box wrong');
    expect(b[1]).toContain('model: box wrong');
  });

  it('shows locked boxes, reader disagreement with both candidates, and the text choice', () => {
    regionVocabularyStore.textChoices = ['vlm_preferred'];
    const el = render({
      region_boxes: [
        wireBox({
          locked: true,
          text: 'TAG-001',
          text_vlm: 'TAG-001',
          text_ocr: 'TAG-008',
          text_disagreement: true,
          text_choice: 'vlm_preferred',
          text_vlm_invalid: 'sequence',
        }),
      ],
    });
    const [b] = blocks(el);
    expect(b).toContain('locked');
    expect(b).toContain('readers disagree');
    expect(b).toContain('vlm: TAG-001');
    expect(b).toContain('ocr: TAG-008');
    expect(b).toContain('Text choice:');
    expect(b).toContain('VLM text rejected:');
  });

  it("renders a rejected box's own reason, worded by the served kind", () => {
    regionVocabularyStore.rejectionReasons = [
      {
        id: 'verifier_no_verdict',
        label: 'Verifier gave no verdict',
        kind: 'needs_human',
        match: 'exact',
        label_template: null,
      },
    ];
    const el = render({
      region_boxes: [
        wireBox({
          state: 'rejected',
          rejection_reason: 'verifier_no_verdict',
        }),
      ],
    });
    const [b] = blocks(el);
    expect(b).toContain('Needs review: Verifier gave no verdict');
  });

  it('flags an incomplete set (region_set_complete === false) and says nothing when complete', () => {
    const incomplete = render({
      region_set_complete: false,
      region_boxes: [wireBox({})],
    });
    expect(incomplete.textContent).toContain('incomplete');
  });

  it('does not flag a complete set', () => {
    const el = render({ region_set_complete: true, region_boxes: [wireBox({})] });
    expect(el.textContent).not.toContain('incomplete');
  });
});

describe('item-level validation', () => {
  it('distinguishes human-validated from auto-confirmed', () => {
    const human = render({
      region_status: 'detected',
      region_validated: true,
      region_auto_confirmed: false,
      region_boxes: [wireBox({})],
    });
    expect(human.textContent).toContain('human validated');
    unmount(instance as never);
    instance = undefined;
    target.remove();
    const auto = render({
      region_status: 'detected',
      region_validated: false,
      region_auto_confirmed: true,
      region_boxes: [wireBox({})],
    });
    expect(auto.textContent).toContain('auto-confirmed (unreviewed)');
  });
});
