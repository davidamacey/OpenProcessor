import { describe, expect, it } from 'vitest';
import { sourceBadge } from './sourceBadge';

describe('sourceBadge', () => {
  it("labels an ingest proposal by the catalog's label, not its raw id", () => {
    const b = sourceBadge('coco_yolo11_proposal', false, 'proposal', 'yolo11 proposal');
    expect(b.text).toBe('yolo11 proposal');
    expect(b.cls).toContain('zinc');
  });

  it('marks an unconfirmed VLM label with ?', () => {
    expect(sourceBadge('vlm', false, 'vlm', 'VLM').text).toBe('VLM?');
    expect(sourceBadge('vlm', true, 'vlm', 'VLM').text).toBe('VLM');
  });

  it('colors by role, so a renamed detector keeps its meaning', () => {
    const a = sourceBadge('det_a_model', true, 'model', 'A');
    const b = sourceBadge('det_b_model', true, 'model', 'B');
    expect(a.cls).toBe(b.cls);
    expect(sourceBadge('human_move', true, 'human', 'Human move').text).toBe('human');
  });

  it('renders an id outside the catalog verbatim and neutral, inferring nothing', () => {
    const b = sourceBadge('vlm_unmatched', false, null, '');
    expect(b.text).toBe('vlm_unmatched');
    expect(b.cls).toContain('zinc');
    expect(sourceBadge('', false, null, '').text).toBe('unlabeled');
  });
});
