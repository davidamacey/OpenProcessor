import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { cleanLogLine } from './logText';
import LogTail from '$lib/components/LogTail.svelte';

describe('cleanLogLine (F-63a)', () => {
  it('strips the erase-line and color escapes the trainer emits', () => {
    expect(cleanLogLine('\x1b[K      1/40  2.1G  1.234: 100%')).toBe(
      '      1/40  2.1G  1.234: 100%',
    );
    expect(cleanLogLine('\x1b[34m\x1b[1mtrain:\x1b[0m Scanning')).toBe('train: Scanning');
  });

  it('keeps what a terminal shows after carriage-return redraws', () => {
    expect(cleanLogLine('epoch 1: 10%\repoch 1: 50%\repoch 1: 100%')).toBe(
      'epoch 1: 100%',
    );
  });

  it('leaves plain text alone', () => {
    expect(cleanLogLine('Results saved to runs/detect/train')).toBe(
      'Results saved to runs/detect/train',
    );
  });
});

describe('LogTail renders cleaned lines', () => {
  let target: HTMLDivElement;
  let instance: ReturnType<typeof mount> | undefined;
  afterEach(() => {
    if (instance) unmount(instance);
    target?.remove();
  });

  it('never shows a raw escape character', async () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(LogTail, {
      target,
      props: {
        jobId: 'run-1',
        active: true,
        intervalMs: 60_000,
        fetcher: async () =>
          ({ job_id: 'run-1', lines: ['\x1b[K  5/40 loss 1.0'] }) as never,
      },
    });
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
    expect(target.textContent).toContain('5/40 loss 1.0');
    expect(target.textContent).not.toContain('\x1b');
  });
});
