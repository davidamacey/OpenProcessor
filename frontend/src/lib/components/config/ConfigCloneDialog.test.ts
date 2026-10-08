/**
 * The shared clone dialog's optional "from another project" picker: the
 * served selectable projects other than the active one, `from_project`
 * handed on only when one is picked.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { projectsStore } from '$stores/projects.svelte';
import { testProject } from '$lib/test/fixtures/projects';
import ConfigCloneDialog from './ConfigCloneDialog.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

beforeEach(() => {
  projectsStore.list = [
    testProject({ slug: 'default' }),
    testProject({ slug: 'alpha' }),
    testProject({ slug: 'building', selectable: false }),
  ];
  projectsStore.current = projectsStore.list[0]!;
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  projectsStore.reset();
});

function render(props: Record<string, unknown>) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ConfigCloneDialog, {
    target,
    props: {
      title: 'Clone x',
      nameLabel: 'New name',
      busy: false,
      error: null,
      report: null,
      onconfirm: () => {},
      oncancel: () => {},
      ...props,
    },
  });
  flushSync();
}

const q = (id: string) => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);

function fill(name: string) {
  const el = q('clone-name') as HTMLInputElement;
  el.value = name;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}
const confirm = () =>
  [...document.querySelectorAll('button')]
    .find((b) => b.textContent?.trim() === 'Clone')!
    .click();

describe('ConfigCloneDialog from-project picker', () => {
  it('offers no picker unless asked', () => {
    render({});
    expect(q('clone-from-project')).toBeNull();
  });

  it('lists the other selectable projects, and hands on none by default', () => {
    const onconfirm = vi.fn();
    render({ offerProjects: true, onconfirm });
    const sel = q('clone-from-project') as HTMLSelectElement;
    expect([...sel.options].map((o) => o.value)).toEqual(['', 'alpha']);
    fill('copy');
    confirm();
    expect(onconfirm).toHaveBeenCalledWith('copy', '', null);
  });

  it('hands on the chosen project slug', () => {
    const onconfirm = vi.fn();
    render({ offerProjects: true, onconfirm });
    const sel = q('clone-from-project') as HTMLSelectElement;
    sel.value = 'alpha';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    fill('copy');
    confirm();
    expect(onconfirm).toHaveBeenCalledWith('copy', '', 'alpha');
  });
});
