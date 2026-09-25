import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import IngestDropZone from './IngestDropZone.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target.remove();
  vi.restoreAllMocks();
});

describe('IngestDropZone', () => {
  it('renders Choose files / Choose folder buttons', () => {
    instance = mount(IngestDropZone, {
      target,
      props: { acceptedExtensions: ['.jpg'], onselect: () => {} },
    });
    flushSync();
    const labels = [...target.querySelectorAll('button')].map((b) =>
      b.textContent?.trim(),
    );
    expect(labels).toContain('Choose files');
    expect(labels).toContain('Choose folder');
  });

  it('filters the file-input selection to accepted extensions before emitting', () => {
    const onselect = vi.fn();
    instance = mount(IngestDropZone, {
      target,
      props: { acceptedExtensions: ['.jpg'], onselect },
    });
    flushSync();
    const fileInput = target.querySelectorAll('input[type=file]')[0];
    expect(fileInput).toBeTruthy();
    const good = new File(['x'], 'a.jpg');
    const bad = new File(['x'], 'b.gif');
    Object.defineProperty(fileInput, 'files', {
      value: [good, bad],
      configurable: true,
    });
    fileInput?.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onselect).toHaveBeenCalledTimes(1);
    const selected = onselect.mock.calls[0]![0] as { relPath: string }[];
    expect(selected.map((f) => f.relPath)).toEqual(['a.jpg']);
  });

  it('disables both buttons when disabled', () => {
    instance = mount(IngestDropZone, {
      target,
      props: { acceptedExtensions: ['.jpg'], onselect: () => {}, disabled: true },
    });
    flushSync();
    for (const b of target.querySelectorAll('button')) {
      expect((b as HTMLButtonElement).disabled).toBe(true);
    }
  });
});
