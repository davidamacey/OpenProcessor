import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { SaveStatus } from '$lib/editor/editorCommands.svelte';
import EditorHeader from './EditorHeader.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(over: Record<string, unknown> = {}) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(EditorHeader, {
    target,
    props: {
      backHref: '/p/default/workspace',
      fileName: 'street.jpg',
      index: 4,
      total: 20,
      status: 'saved' as SaveStatus,
      message: null,
      ...over,
    },
  });
  flushSync();
}

const state = () =>
  target.querySelector('[data-testid="save-state"]')!.textContent?.trim();

describe('EditorHeader', () => {
  it('shows the file, the way back and the served position', () => {
    render();
    expect(target.querySelector('h1')!.textContent).toBe('street.jpg');
    expect(
      target.querySelector('[data-testid="editor-back"]')!.getAttribute('href'),
    ).toBe('/p/default/workspace');
    expect(
      target.querySelector('[data-testid="editor-position"]')!.textContent?.trim(),
    ).toBe('5 of 20');
  });

  it('shows no position for an image outside the list', () => {
    render({ index: null, total: null });
    expect(target.querySelector('[data-testid="editor-position"]')).toBeNull();
  });

  it.each([
    ['saved', 'Saved'],
    ['saving', 'Saving…'],
    ['conflict', 'Changed by someone else, reloaded'],
    ['failed', 'Not saved'],
  ] as const)('shows %s as "%s"', (status, text) => {
    render({ status });
    expect(state()).toBe(text);
  });

  it('shows the served words under a conflict or a failed save, not otherwise', () => {
    render({ status: 'conflict', message: 'The box changed. Reload.' });
    expect(target.querySelector('[role="alert"]')!.textContent?.trim()).toBe(
      'The box changed. Reload.',
    );
    unmount(instance!);
    target.remove();
    render({ status: 'saved', message: 'old message' });
    expect(target.querySelector('[role="alert"]')).toBeNull();
  });
});
