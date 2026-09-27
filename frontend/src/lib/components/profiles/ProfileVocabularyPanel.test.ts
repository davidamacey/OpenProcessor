/**
 * The read-only vocabulary panel, mounted: every served list renders with
 * its served labels and facts (the configured OCR model, a promoted
 * detector's project and class mapping, the segmenter's cap and floor, a
 * VLM that sends images out), and an empty segmenter list says so.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { vocabularyFixture } from '$lib/test/fixtures/regionProfiles';
import type { ConfigVocabulary } from '$lib/types_profiles';
import ProfileVocabularyPanel from './ProfileVocabularyPanel.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(vocab: ConfigVocabulary) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ProfileVocabularyPanel, { target, props: { vocab } });
  flushSync();
}

const q = (id: string) => target.querySelector(`[data-testid="${id}"]`);

describe('ProfileVocabularyPanel', () => {
  it('renders the served detectors with their served facts', () => {
    render(vocabularyFixture());
    const det = q('vocab-detectors')!.textContent!;
    expect(det).toContain('tag_detector_v1 (promoted)');
    expect(det).toContain('promoted');
    expect(det).toContain('project default');
    expect(det).toContain('1 classes map');
    expect(det).toContain('item_detector_base');
  });

  it('marks the configured OCR model and lists every OCR list', () => {
    render(vocabularyFixture());
    expect(q('vocab-ocr-det')!.textContent).toContain('configured for this role');
    expect(q('vocab-ocr-det')!.textContent).toContain('ocr_det_v2');
    expect(q('vocab-ocr-rec')!.textContent).toContain('ocr_rec_v1');
  });

  it('shows the segmenter cap and floor as served', () => {
    render(vocabularyFixture());
    const row = q('segmenter-row')!.textContent!;
    expect(row).toContain('segmenter_v1');
    expect(row).toContain('128');
    expect(row).toContain('0.5');
  });

  it('text modes and VLM endpoints as served', () => {
    const v = vocabularyFixture();
    v.vlm.endpoints[0]!.sends_images_externally = true;
    render(v);
    expect(q('vocab-text-modes')!.textContent).toContain('Off (region has no text)');
    expect(q('vocab-text-modes')!.textContent).toContain('needs OCR');
    expect(q('vocab-vlm')!.textContent).toContain('sends images externally');
    expect(q('vocab-vlm')!.textContent).toContain('example/vision-model');
  });

  it('no segmenter says so', () => {
    render({ ...vocabularyFixture(), segmenters: [] });
    expect(q('segmenter-status')!.textContent).toContain('No segmenter is configured');
  });
});
