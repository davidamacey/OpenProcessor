/**
 * Mount a component for a test and unmount every one in `afterEach`.
 */
import { afterEach } from 'vitest';
import { createRawSnippet, flushSync, mount, unmount, type Component } from 'svelte';

const live: Array<{ instance: ReturnType<typeof mount>; target: HTMLElement }> = [];

afterEach(() => {
  while (live.length) {
    const { instance, target } = live.pop()!;
    unmount(instance);
    target.remove();
  }
});

/** A snippet prop (`children`, `actions`, ...) that renders `markup`, which must be one element. */
export function snip(markup: string) {
  return createRawSnippet(() => ({ render: () => markup }));
}

export function mountUi<P extends Record<string, unknown>>(
  component: Component<any>, // eslint-disable-line @typescript-eslint/no-explicit-any
  props: P = {} as P,
): HTMLElement {
  const target = document.createElement('div');
  document.body.appendChild(target);
  const instance = mount(component, { target, props });
  live.push({ instance, target });
  flushSync();
  return target;
}
