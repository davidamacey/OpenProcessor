/**
 * Every served issue list is keyed by the served `id` (`code[:subject]`, `#2`
 * on a repeat), not by position or `code`: dropping the first of two issues
 * that share a code must keep the second one's DOM node.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import CombineIssueList from './combine/CombineIssueList.svelte';
import ConfigIssueList from './config/ConfigIssueList.svelte';
import DatasetIssueList from './datasets/DatasetIssueList.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function setup() {
  target = document.createElement('div');
  document.body.appendChild(target);
}

describe('issue lists key rows by the served id', () => {
  it('CombineIssueList', () => {
    setup();
    const a = { code: 'unmapped_class', id: 'unmapped_class:a:x', message: 'm1' };
    const b = { code: 'unmapped_class', id: 'unmapped_class:a:x#2', message: 'm2' };
    const props = $state({ issues: [a, b] });
    instance = mount(CombineIssueList, { target, props });
    flushSync();
    const second = target.querySelectorAll('[data-testid="combine-issue"]')[1];
    props.issues = [b];
    flushSync();
    expect(target.querySelector('[data-testid="combine-issue"]')).toBe(second);
  });

  it('ConfigIssueList', () => {
    setup();
    const mk = (id: string, message: string) => ({
      code: 'field_invalid',
      id,
      severity: 'error' as const,
      field: 'x',
      message,
      detail: {},
      bypassable: false,
    });
    const a = mk('field_invalid:x', 'm1');
    const b = mk('field_invalid:x#2', 'm2');
    const props = $state({ issues: [a, b] });
    instance = mount(ConfigIssueList, { target, props });
    flushSync();
    const second = target.querySelectorAll('[data-testid="config-issue"]')[1];
    props.issues = [b];
    flushSync();
    expect(target.querySelector('[data-testid="config-issue"]')).toBe(second);
  });

  it('DatasetIssueList', () => {
    setup();
    const mk = (id: string, message: string) => ({
      code: 'label_file_missing',
      id,
      severity: 'warning' as const,
      blocking: false,
      bypassable: false,
      message,
      count: 1,
      samples: [],
    });
    const a = mk('label_file_missing', 'm1');
    const b = mk('label_file_missing#2', 'm2');
    const props = $state({ issues: [a, b], catalog: [] });
    instance = mount(DatasetIssueList, { target, props });
    flushSync();
    const second = target.querySelectorAll('[data-code]')[1];
    props.issues = [b];
    flushSync();
    expect(target.querySelector('[data-code]')).toBe(second);
  });
});
