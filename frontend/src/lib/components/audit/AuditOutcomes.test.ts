import { describe, expect, it } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import { projectsStore } from '$stores/projects.svelte';
import { testProject } from '$lib/test/fixtures/projects';
import AuditOutcomes from './AuditOutcomes.svelte';

describe('AuditOutcomes', () => {
  it('links every served outcome count to Detections on that outcome', () => {
    projectsStore.current = testProject({ slug: 'demo' });
    const el = mountUi(AuditOutcomes, {
      outcomes: { agree: 44, detector_wrong: 5, vlm_wrong: 2, both_wrong: 1 },
    });
    const links = [...el.querySelectorAll('a')].map((a) => [
      a.getAttribute('data-outcome'),
      a.textContent!.replace(/\s+/g, ' ').trim(),
      a.getAttribute('href'),
    ]);
    expect(links).toEqual([
      ['agree', 'Agree: 44', '/p/demo/workspace?view=detections&audit_outcome=agree'],
      [
        'detector_wrong',
        'Detector wrong: 5',
        '/p/demo/workspace?view=detections&audit_outcome=detector_wrong',
      ],
      [
        'vlm_wrong',
        'VLM wrong: 2',
        '/p/demo/workspace?view=detections&audit_outcome=vlm_wrong',
      ],
      [
        'both_wrong',
        'Both wrong: 1',
        '/p/demo/workspace?view=detections&audit_outcome=both_wrong',
      ],
    ]);
  });
});
