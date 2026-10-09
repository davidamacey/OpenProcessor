import { describe, expect, it } from 'vitest';
import { confirmationLabel, sourceBadge } from './sourceBadge';

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
    expect(unconfirmed.text).toBe('VLM suggestion');

    const confirmed = sourceBadge('vlm', true, 'vlm', 'VLM');
    expect(confirmed.text).toBe('VLM');
    expect(confirmed.unvalidated).toBe(false);
  });

  it('colors by role, so a renamed detector keeps its meaning', () => {
    const a = sourceBadge('det_a_model', true, 'model', 'A');
    const b = sourceBadge('det_b_model', true, 'model', 'B');
    expect(a.cls).toBe(b.cls);
    expect(sourceBadge('human_move', true, 'human', 'Human move').text).toBe(
      'Human-confirmed',
    );
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

describe('confirmation wording (#119)', () => {
  it('names what a label is worth from the role and the validated flag alone', () => {
    expect(confirmationLabel('human', true)).toBe('Human-confirmed');
    expect(confirmationLabel('vlm', false)).toBe('VLM suggestion');
    expect(confirmationLabel('cluster', true)).toBe('Auto-validated');
  });

  it('has no wording for a role that is not one of the three', () => {
    expect(confirmationLabel('vlm', true)).toBeNull();
    expect(confirmationLabel('cluster', false)).toBeNull();
    expect(confirmationLabel('model', true)).toBeNull();
    expect(confirmationLabel('proposal', false)).toBeNull();
    expect(confirmationLabel(null, true)).toBeNull();
  });

  it('badges an unvalidated VLM label as a suggestion and a validated cluster label as auto-validated', () => {
    expect(sourceBadge('vlm', false, 'vlm', 'VLM').text).toBe('VLM suggestion');
    const auto = sourceBadge('cluster_agree', true, 'cluster', 'Cluster agreement');
    expect(auto.text).toBe('Auto-validated');
    expect(sourceBadge('cluster_agree', false, 'cluster', 'Cluster agreement').text).toBe(
      'Cluster agreement',
    );
  });
});
