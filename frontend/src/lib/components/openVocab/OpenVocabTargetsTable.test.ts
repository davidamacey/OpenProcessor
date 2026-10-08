/**
 * The targets table, mounted: one row per target in order, served cells,
 * the discovery hint for an empty class name, the served issue under the
 * cell its path names, advanced cells behind a per-row expander, row
 * actions, and no controls when read-only.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { bodyFixture, errorReport, issue, schemaFixture } from '$lib/openVocab/fixtures';
import { rowsByScope } from '$lib/openVocab/openVocabFields';
import OpenVocabTargetsTable from './OpenVocabTargetsTable.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(over: Record<string, unknown> = {}) {
  const fns = {
    onadd: vi.fn(),
    onremove: vi.fn(),
    onmove: vi.fn(),
    onchange: vi.fn(),
  };
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(OpenVocabTargetsTable, {
    target,
    props: {
      targets: bodyFixture().targets!,
      rows: rowsByScope(schemaFixture()).target,
      report: null,
      classNames: ['widget', 'gadget'],
      maxEnabledTargets: 8,
      maxEnabledTargetsCeiling: 16,
      ...fns,
      ...over,
    },
  });
  flushSync();
  return fns;
}

const rows = () => [...target.querySelectorAll('[data-testid="target-row"]')];

describe('OpenVocabTargetsTable', () => {
  it('renders one row per target in served order with served cell labels', () => {
    render();
    expect(rows()).toHaveLength(2);
    expect(rows()[0]!.textContent).toContain('blue widget');
    expect(rows()[1]!.textContent).toContain('cracked widget');
    expect(rows()[0]!.textContent).toContain('Minimum score');
    expect(target.querySelector('[data-testid="targets-facts"]')!.textContent).toContain(
      '2 enabled of 2',
    );
    expect(target.querySelector('[data-testid="targets-facts"]')!.textContent).toContain(
      'ceiling 16',
    );
  });

  it('autocompletes class names but takes any text', () => {
    const f = render();
    const opts = [...target.querySelectorAll('datalist option')].map((o) =>
      o.getAttribute('value'),
    );
    expect(opts).toEqual(['widget', 'gadget']);
    const input = rows()[0]!.querySelector('input[list]') as HTMLInputElement;
    input.value = 'anything at all';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    expect(f.onchange).toHaveBeenCalledWith(0, 'class_name', 'anything at all');
  });

  it('explains discovery mode only for an empty class name', () => {
    render();
    expect(rows()[0]!.querySelector('[data-testid="discovery-hint"]')).toBeNull();
    expect(
      rows()[1]!.querySelector('[data-testid="discovery-hint"]')!.textContent,
    ).toContain('discovery mode');
  });

  it('puts a served issue under the cell its path names and nowhere else', () => {
    render({
      report: errorReport(
        issue({ field: 'targets[1].prompt', message: 'Prompt is too vague.' }),
        issue({ field: 'targets[1].class_name', message: 'Unknown parent class.' }),
      ),
    });
    expect(rows()[1]!.textContent).toContain('Prompt is too vague.');
    expect(rows()[1]!.querySelector('[data-testid="cell-issue"]')!.textContent).toContain(
      'Unknown parent class.',
    );
    expect(rows()[0]!.textContent).not.toContain('Prompt is too vague.');
    expect(rows()[0]!.textContent).not.toContain('Unknown parent class.');
  });

  it('shows advanced cells only after the row expander is opened', () => {
    render();
    expect(rows()[0]!.querySelector('[data-testid="target-advanced"]')).toBeNull();
    (
      rows()[0]!.querySelector('[data-testid="target-advanced-toggle"]') as HTMLElement
    ).click();
    flushSync();
    const adv = rows()[0]!.querySelector('[data-testid="target-advanced"]')!;
    expect(adv.textContent).toContain('Most instances');
    expect(adv.textContent).toContain('Only inside these item classes');
  });

  it('emits row actions with the row index', () => {
    const f = render();
    (rows()[1]!.querySelector('[data-testid="target-up"]') as HTMLElement).click();
    (rows()[0]!.querySelector('[data-testid="target-remove"]') as HTMLElement).click();
    (target.querySelector('[data-testid="target-add"]') as HTMLElement).click();
    expect(f.onmove).toHaveBeenCalledWith(1, -1);
    expect(f.onremove).toHaveBeenCalledWith(0);
    expect(f.onadd).toHaveBeenCalledTimes(1);
    expect(
      (rows()[0]!.querySelector('[data-testid="target-up"]') as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  it('offers no editing controls when read-only', () => {
    render({ readonly: true });
    expect(target.querySelector('[data-testid="target-add"]')).toBeNull();
    expect(target.querySelector('[data-testid="target-remove"]')).toBeNull();
  });
});
