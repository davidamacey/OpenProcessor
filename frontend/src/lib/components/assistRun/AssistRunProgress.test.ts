import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import AssistRunProgress from '$components/assistRun/AssistRunProgress.svelte';
import { assistRun } from '$lib/test/fixtures/assist';
import type { AssistRunState } from '$lib/types_assist';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(run: AssistRunState) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(AssistRunProgress, { target, props: { run } });
  flushSync();
}

const q = (id: string) =>
  target.querySelector(`[data-testid="${id}"]`) as HTMLElement | null;

describe('AssistRunProgress', () => {
  it('says the run is waiting for the region worker, with the served segment counts', () => {
    render(
      assistRun('waiting_region', {
        result: { segment: { queued: 12, done: 5, pending: 7, failed: 0 } },
      }),
    );
    expect(q('assist-status')?.textContent).toContain('Waiting for the region worker');
    expect(q('assist-waiting-region')?.textContent).toContain('5 of 12 done');
  });

  it('lists the segment, verify and attribute outcomes as served when the run ends', () => {
    render(
      assistRun('completed_with_errors', {
        result: {
          segment: { queued: 12, done: 10, pending: 0, failed: 2 },
          verify: { boxes: 10, accepted: 8, rejected: 2, locked_skipped: 1 },
          attributes: { items_written: 9, invalid_values: 3 },
        },
      }),
    );
    const text = (q('assist-result')?.textContent ?? '').replace(/\s+/g, ' ');
    expect(text).toContain('Segment: 12 queued, 10 done, 0 still pending, 2 failed');
    expect(text).toContain('Verify: 10 boxes judged, 8 accepted, 2 rejected, 1 locked');
    expect(text).toContain('Attributes: 9 items');
    expect(q('assist-status')?.textContent).toContain('Completed with errors');
  });

  it('shows the served error of a failed run', () => {
    render(assistRun('failed', { error: 'the VLM endpoint is unreachable' }));
    expect(q('assist-run-error')?.textContent).toBe('the VLM endpoint is unreachable');
  });
});
