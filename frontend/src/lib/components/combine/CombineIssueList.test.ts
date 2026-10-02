import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import CombineIssueList from './CombineIssueList.svelte';
import type { CombineIssue } from '$lib/types_combine';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

function render(issues: CombineIssue[]) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CombineIssueList, { target, props: { issues } });
  flushSync();
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

describe('CombineIssueList', () => {
  it('renders nothing for no issues', () => {
    render([]);
    expect(target.querySelector('[data-testid="combine-issues"]')).toBeNull();
  });

  it('shows severity, the served message, project chip, code and a detail disclosure', () => {
    render([
      {
        code: 'unmapped_class',
        severity: 'error',
        project: 'widgets-a',
        message: 'class gadget has no mapping',
        detail: { class: 'gadget', count: 2 },
      },
      { code: 'label_conflicts', severity: 'warning', message: '1 boxes disagree' },
      { code: 'slug_taken', detail: {} },
    ]);
    const items = [...target.querySelectorAll('[data-testid="combine-issue"]')];
    expect(items).toHaveLength(3);
    expect(items[0]!.getAttribute('data-severity')).toBe('error');
    expect(items[0]!.textContent).toContain('class gadget has no mapping');
    expect(items[0]!.textContent).toContain('widgets-a');
    expect(items[0]!.textContent).toContain('unmapped_class');
    expect(items[0]!.querySelector('pre')?.textContent).toContain('"count": 2');
    expect(items[1]!.getAttribute('data-severity')).toBe('warning');
    expect(items[1]!.querySelector('details')).toBeNull();
    // A severity-less issue is an error, as the contract defaults it.
    expect(items[2]!.getAttribute('data-severity')).toBe('error');
    // An empty detail object is not worth a disclosure.
    expect(items[2]!.querySelector('details')).toBeNull();
  });
});
