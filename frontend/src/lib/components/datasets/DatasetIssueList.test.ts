/**
 * Served issues render grouped by served severity, headed by the served
 * catalog label, with the served message, count, blocking flag and
 * samples (W10.4) — the same component shows a 422 `import_blocked`'s
 * issues.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import DatasetIssueList from './DatasetIssueList.svelte';
import { formatsFixture } from '$lib/test/fixtures/datasetImport';
import type { DatasetIssue } from '$lib/types_import';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

function render(issues: DatasetIssue[]) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(DatasetIssueList, {
    target,
    props: { issues, catalog: formatsFixture().issues },
  });
  flushSync();
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const blocking: DatasetIssue = {
  code: 'test_split_changed',
  id: 'test_split_changed',
  severity: 'error',
  blocking: true,
  bypassable: true,
  message: 'TEST_FROZEN.json no longer verifies.',
  count: 3,
  samples: [{ file: 'labels/test/a.txt', line: 4, detail: {} }],
};
const warning: DatasetIssue = {
  code: 'label_file_missing',
  id: 'label_file_missing',
  severity: 'warning',
  blocking: false,
  bypassable: false,
  message: '1 image has no label file.',
  count: 1,
  samples: [],
};

describe('DatasetIssueList', () => {
  it('groups by served severity with the served catalog label and message', () => {
    render([warning, blocking]);
    const errors = target.querySelector('[data-testid="dataset-issues-error"]')!;
    const warnings = target.querySelector('[data-testid="dataset-issues-warning"]')!;
    expect(target.querySelector('[data-testid="dataset-issues-info"]')).toBeNull();
    expect(errors.textContent).toContain('The frozen test split changed');
    expect(errors.textContent).toContain('TEST_FROZEN.json no longer verifies.');
    expect(errors.textContent).toContain('blocks the import (can be forced)');
    expect(errors.textContent).toContain('labels/test/a.txt:4');
    expect(warnings.textContent).toContain('Images without a label file');
    expect(warnings.textContent).not.toContain('blocks the import');
  });

  it('falls back to the code when the catalog has no label, and says when there are none', () => {
    render([{ ...warning, code: 'new_code_x' }]);
    expect(target.textContent).toContain('new_code_x');
    unmount(instance!);
    target.remove();
    render([]);
    expect(target.querySelector('[data-testid="dataset-issues-none"]')).not.toBeNull();
  });
  it('renders two served issues that share a code (a Map row with no target plus an unmapped class)', () => {
    const unmapped: DatasetIssue = {
      code: 'class_unmapped',
      id: 'class_unmapped',
      severity: 'error',
      blocking: true,
      bypassable: false,
      message: 'A dataset class with boxes has no mapping',
      count: 1,
      samples: [],
    };
    render([unmapped, { ...unmapped, count: 2 }]);
    expect(target.querySelectorAll('li[data-code="class_unmapped"]').length).toBe(2);
  });
});
