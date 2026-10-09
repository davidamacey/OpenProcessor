import { describe, expect, it } from 'vitest';
import { sourceBadge } from './sourceBadge';

describe('sourceBadge', () => {
  it("labels an ingest proposal by the catalog's label, not its raw id", () => {
    const b = sourceBadge('coco_yolo11_proposal', false, 'proposal', 'yolo11 proposal');
    expect(b.text).toBe('yolo11 proposal');
    expect(b.cls).toContain('zinc');
  });

  it('marks an unconfirmed VLM label with the unvalidated flag, not a literal "?" (p4, 2026-09-24 interactive pass)', () => {
    // p4: a bare "?" suffixed straight into the badge text read as a
    // question ("Labeled by the VLM?"). The badge text itself no longer
    // contains "?" — callers render the real explanation via `title`
    // using the `unvalidated` flag instead.
    const unconfirmed = sourceBadge('vlm', false, 'vlm', 'VLM');
    expect(unconfirmed.text).not.toContain('?');
    expect(unconfirmed.unvalidated).toBe(true);

    const confirmed = sourceBadge('vlm', true, 'vlm', 'VLM');
    expect(confirmed.text).toBe('VLM');
    expect(confirmed.unvalidated).toBe(false);
  });

  it('colors by role, so a renamed detector keeps its meaning', () => {
    const a = sourceBadge('det_a_model', true, 'model', 'A');
    const b = sourceBadge('det_b_model', true, 'model', 'B');
    expect(a.cls).toBe(b.cls);
    expect(sourceBadge('human_move', true, 'human', 'Human move').text).toBe('human');
  });

  it('gives the open_vocab role its own sky tone, labelled by the catalog', () => {
    const b = sourceBadge(
      'open_vocab_target',
      false,
      'open_vocab',
      'Open-vocabulary target',
    );
    expect(b.text).toBe('Open-vocabulary target');
    expect(b.cls).toContain('sky');
    expect(b.cls).not.toBe(sourceBadge('x', false, 'proposal', 'x').cls);
    expect(b.unvalidated).toBe(false);
  });

  it('renders an id outside the catalog verbatim and neutral, inferring nothing', () => {
    const b = sourceBadge('vlm_unmatched', false, null, '');
    expect(b.text).toBe('vlm_unmatched');
    expect(b.cls).toContain('zinc');
    expect(sourceBadge('', false, null, '').text).toBe('unlabeled');
  });
});
