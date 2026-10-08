import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import IngestDetectorCard from './IngestDetectorCard.svelte';
import { servedIngestConfig } from '$lib/test/fixtures/ingestConfig';
import type { IngestDetectorInfo } from '$lib/types_detector';

let target: HTMLElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const DETECTOR: IngestDetectorInfo = {
  model: 'widget_detector_v1',
  version: '2',
  input_size: 640,
  assigns_class: false,
  confidence_floor_applies: false,
  n_labels: 2,
  labels: [
    { class_id: 0, name: 'widget', slug: 'widget' },
    { class_id: 1, name: 'tag label', slug: 'tag_label' },
  ],
};

function render(over: Parameters<typeof servedIngestConfig>[0]) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(IngestDetectorCard, {
    target,
    props: { config: servedIngestConfig(over) },
  });
  flushSync();
}

describe('IngestDetectorCard', () => {
  it('reads "No detector reported." for a null detector', () => {
    render({ detector: null });
    expect(target.textContent).toContain('No detector reported.');
    expect(target.querySelector('table')).toBeNull();
  });

  it('reads "No detector reported." when the key is absent', () => {
    render({ detector: undefined });
    expect(target.textContent).toContain('No detector reported.');
  });

  it('shows the served model, version, input size, class assignment and label count', () => {
    render({ detector: DETECTOR });
    const text = target.textContent!;
    expect(text).toContain('widget_detector_v1');
    expect(text).toContain('640');
    expect(text).toContain('2 labels');
    expect(
      target.querySelector('[data-testid="detector-assigns-class"]')!.textContent,
    ).toContain('no');
  });

  it('lists the label table name to slug inside a collapsed details', () => {
    render({ detector: DETECTOR });
    const details = target.querySelector('details')!;
    expect(details.open).toBe(false);
    const rows = [...details.querySelectorAll('tbody tr')].map((r) =>
      [...r.querySelectorAll('td')].map((c) => c.textContent!.trim()).join('|'),
    );
    expect(rows).toEqual(['0|widget|widget', '1|tag label|tag_label']);
  });

  it('summarises the served policy and links to the policy page', () => {
    render({
      detector: DETECTOR,
      policy: {
        embedding: { mode: 'lazy' },
        detect: { min_confidence: 0.4 },
        revision: 3,
      },
    });
    const line = target.querySelector('[data-testid="detector-policy-summary"]')!;
    expect(line.textContent).toContain('Embedding: lazy');
    expect(line.textContent).toContain('min confidence 0.4');
    const a = target.querySelector('a[href$="/settings/ingest-policy"]');
    expect(a).not.toBeNull();
  });

  it('shows no policy line when the config carries none', () => {
    render({ detector: DETECTOR });
    expect(target.querySelector('[data-testid="detector-policy-summary"]')).toBeNull();
  });
});
