import { describe, expect, it } from 'vitest';
import { mapRawCrop } from '$lib/api';
import { makeItem } from '$lib/test/makeItem';
import { detectorClassText, percentText } from './labelConfirmation';

describe('detectorClassText', () => {
  it('shows the detector class and score beside a VLM-sourced label', () => {
    expect(
      detectorClassText('vlm', {
        detector_class_name: 'widget_b',
        detector_confidence: 0.724,
      }),
    ).toBe('widget_b 72%');
  });

  it('shows the class alone when the detector gave no score', () => {
    expect(
      detectorClassText('vlm', {
        detector_class_name: 'widget_b',
        detector_confidence: null,
      }),
    ).toBe('widget_b');
  });

  it('shows nothing for any other label source', () => {
    const crop = { detector_class_name: 'widget_b', detector_confidence: 0.5 };
    for (const role of [
      'human',
      'model',
      'cluster',
      'proposal',
      'vlm_unmatched',
      null,
    ] as const) {
      expect(detectorClassText(role, crop)).toBeNull();
    }
  });

  it('shows nothing when the item carries no detector answer', () => {
    expect(
      detectorClassText('vlm', { detector_class_name: null, detector_confidence: 0.9 }),
    ).toBeNull();
    expect(detectorClassText('vlm', {})).toBeNull();
  });
});

describe('percentText', () => {
  it('formats a served ratio and prints a dash for none', () => {
    expect(percentText(0.5)).toBe('50%');
    expect(percentText(0.8333, 1)).toBe('83.3%');
    expect(percentText(null)).toBe('—');
    expect(percentText(undefined)).toBe('—');
  });
});

describe('mapRawCrop carries the detector class', () => {
  it('maps the three served detector keys onto the crop', () => {
    const crop = mapRawCrop(
      makeItem({
        detector_class_name: 'widget_h',
        detector_class_id: 8,
        detector_confidence: 0.72,
      }),
    );
    expect(crop.detector_class_name).toBe('widget_h');
    expect(crop.detector_class_id).toBe(8);
    expect(crop.detector_confidence).toBe(0.72);
  });

  it('maps absent keys to null, never a stand-in', () => {
    const crop = mapRawCrop(
      makeItem({
        detector_class_name: null,
        detector_class_id: null,
        detector_confidence: null,
      }),
    );
    expect(crop.detector_class_name).toBeNull();
    expect(crop.detector_confidence).toBeNull();
  });
});
