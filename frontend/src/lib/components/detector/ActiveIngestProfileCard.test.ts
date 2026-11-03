import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ActiveIngestProfileCard from './ActiveIngestProfileCard.svelte';
import type { ActiveConfigResponse } from '$lib/types_config';
import type { IngestDetectorInfo } from '$lib/types_detector';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const NONE: ActiveConfigResponse = {
  axis: 'ingest_profile',
  active: { name: null, revision: null },
  source: 'env',
  activated_at: null,
  previous: null,
  config_revision: 1,
  stale: false,
  applied: [],
};

const DETECTOR: IngestDetectorInfo = {
  model: 'yolo11_small_trt',
  version: '1',
  input_size: 640,
  assigns_class: false,
  confidence_floor_applies: false,
  n_labels: 0,
  labels: [],
};

function render(detector: IngestDetectorInfo | null): string {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ActiveIngestProfileCard, {
    target,
    props: { profile: NONE, error: null, detector },
  });
  flushSync();
  return target.querySelector('[data-testid="active-ingest-profile-none"]')!.textContent!;
}

describe('ActiveIngestProfileCard with no active profile', () => {
  it('says ingest uses the deployment detector, never that it is refused', () => {
    const text = render(DETECTOR);
    expect(text).toContain('deployment');
    expect(text).toContain('yolo11_small_trt');
    expect(text).not.toContain('refused');
  });

  it('still states the fallback when no detector is served', () => {
    const text = render(null);
    expect(text).toContain('deployment');
    expect(text).not.toContain('refused');
  });
});
