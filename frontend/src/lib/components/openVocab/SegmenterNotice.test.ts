/**
 * The segmenter fact the open-vocabulary pages show: every state reads as
 * served (nothing is assumed), nothing renders for an unread fact, and the
 * notice never carries a control that could hide the editor.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import SegmenterNotice from './SegmenterNotice.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(segmenter: { configured: boolean; reachable: boolean } | null) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(SegmenterNotice, { target, props: { segmenter } });
  flushSync();
  return target.querySelector('[data-testid="segmenter-notice"]');
}

describe('SegmenterNotice', () => {
  it('renders nothing before the fact is read', () => {
    expect(render(null)).toBeNull();
  });

  it('says a configured, reachable segmenter is ready', () => {
    const el = render({ configured: true, reachable: true })!;
    expect(el.textContent).toContain('ready');
    expect(el.getAttribute('data-state')).toBe('ready');
  });

  it('warns when it is configured but not reachable', () => {
    const el = render({ configured: true, reachable: false })!;
    expect(el.getAttribute('data-state')).toBe('unreachable');
    expect(el.textContent).toContain('not reachable');
  });

  it('warns when none is configured', () => {
    const el = render({ configured: false, reachable: false })!;
    expect(el.getAttribute('data-state')).toBe('not_configured');
    expect(el.textContent).toContain('No segmenter is configured');
  });

  it('has no button or link that could hide the editor', () => {
    const el = render({ configured: false, reachable: false })!;
    expect(el.querySelector('button, a')).toBeNull();
  });
});
