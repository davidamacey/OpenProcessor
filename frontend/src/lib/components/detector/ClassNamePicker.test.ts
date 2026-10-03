import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import ClassNamePicker from './ClassNamePicker.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(props: {
  value: string[];
  options: string[];
  onchange?: (n: string[]) => void;
}) {
  target = document.createElement('div');
  document.body.appendChild(target);
  const onchange = props.onchange ?? vi.fn();
  instance = mount(ClassNamePicker, {
    target,
    props: { value: props.value, options: props.options, label: 'Classes', onchange },
  });
  flushSync();
  return onchange;
}

function typeAndAdd(text: string): void {
  const input = target.querySelector<HTMLInputElement>('input.input')!;
  input.value = text;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
  target.querySelector<HTMLButtonElement>('button')!.click();
}

describe('ClassNamePicker', () => {
  it('adds a name typed twice once', () => {
    const onchange = render({ value: [], options: ['car'] });
    typeAndAdd('foo, foo');
    expect(onchange).toHaveBeenCalledWith(['foo']);
  });

  it('control: distinct names are both added', () => {
    const onchange = render({ value: [], options: ['car'] });
    typeAndAdd('bar, baz');
    expect(onchange).toHaveBeenCalledWith(['bar', 'baz']);
  });

  it('mounts with repeated option names (detector id gaps are served as empty names)', () => {
    render({ value: [], options: ['person', '', 'car', ''] });
    expect(target.querySelectorAll('input[type="checkbox"]').length).toBe(4);
  });

  it('mounts when the value already holds a name twice', () => {
    render({ value: ['foo', 'foo'], options: ['car'] });
    expect(target.querySelectorAll('button.chip').length).toBe(2);
  });
});
