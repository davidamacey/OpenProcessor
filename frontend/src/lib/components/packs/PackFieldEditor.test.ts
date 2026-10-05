/**
 * One pack field, mounted: the served label/help/chips render, a text
 * field edits through a textarea, a map field through key/value rows, and
 * only the served issues handed to it show.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { issue, schemaFixture } from '$lib/test/fixtures/promptPacks';
import type { ValidationIssue } from '$lib/types_config';
import type { PackFieldValue, PackSchemaField } from '$lib/types_packs';
import PackFieldEditor from './PackFieldEditor.svelte';
import ConfigIssueList from '$components/config/ConfigIssueList.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function field(id: string): PackSchemaField {
  return schemaFixture().fields.find((f) => f.field === id)!;
}

function render(props: {
  field: PackSchemaField;
  value: PackFieldValue | undefined;
  issues?: ValidationIssue[];
  readonly?: boolean;
  onchange?: (v: PackFieldValue) => void;
}) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(PackFieldEditor, {
    target,
    props: { issues: [], onchange: () => {}, ...props },
  });
  flushSync();
}

function type(el: HTMLInputElement | HTMLTextAreaElement, value: string) {
  el.value = value;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}

describe('PackFieldEditor', () => {
  it('renders the served label, help, placeholder and reply-key chips, and used_by', () => {
    render({ field: field('combined_user_template'), value: '{class_block}' });
    expect(target.textContent).toContain('Classify + verify: user message');
    expect(target.textContent).toContain(
      'Classifies the item and judges each numbered box.',
    );
    const chips = [...target.querySelectorAll('[data-testid="placeholder-chip"]')];
    expect(
      chips.map((c) => [c.textContent?.trim(), c.getAttribute('data-required')]),
    ).toEqual([
      ['{class_block} required', 'true'],
      ['{region_block} required', 'true'],
    ]);
    const keys = [...target.querySelectorAll('[data-testid="reply-key-chip"]')].map((c) =>
      c.textContent?.trim(),
    );
    expect(keys).toEqual([
      'class_id',
      'class_confidence',
      'region_boxes',
      'box',
      'region_set_complete optional',
    ]);
    expect(target.textContent).toContain('used by: detection_worker_combined_verify');
  });

  it('marks an allowed-but-not-required placeholder as optional', () => {
    const f = {
      ...field('class_user_template'),
      allowed_placeholders: ['class_names_csv', 'extra'],
    };
    render({ field: f, value: '' });
    const chips = [...target.querySelectorAll('[data-testid="placeholder-chip"]')];
    expect(chips.map((c) => c.getAttribute('data-required'))).toEqual(['true', 'false']);
  });

  it('a text field reports every edit', () => {
    const onchange = vi.fn();
    render({ field: field('class_system'), value: 'abc', onchange });
    const ta = target.querySelector('textarea')!;
    expect(ta.value).toBe('abc');
    type(ta, 'abcd');
    expect(onchange).toHaveBeenCalledWith('abcd');
  });

  it('a map field edits keys and values, adds and removes rows', () => {
    const onchange = vi.fn();
    render({ field: field('synonyms'), value: { a: 'widget', b: 'gadget' }, onchange });
    const inputs = [
      ...target.querySelectorAll<HTMLInputElement>('[data-testid="pack-map"] input'),
    ];
    expect(inputs.map((i) => i.value)).toEqual(['a', 'widget', 'b', 'gadget']);
    type(inputs[0]!, 'z');
    expect(onchange).toHaveBeenLastCalledWith({ z: 'widget', b: 'gadget' });
    type(inputs[3]!, 'thing');
    expect(onchange).toHaveBeenLastCalledWith({ a: 'widget', b: 'thing' });
    const remove = [...target.querySelectorAll('button')].find(
      (b) => b.textContent === 'Remove',
    )!;
    remove.click();
    expect(onchange).toHaveBeenLastCalledWith({ b: 'gadget' });
    const add = [...target.querySelectorAll('button')].find(
      (b) => b.textContent === 'Add entry',
    )!;
    add.click();
    expect(onchange).toHaveBeenLastCalledWith({ a: 'widget', b: 'gadget', '': '' });
  });

  function listInputs(): HTMLInputElement[] {
    return [
      ...target.querySelectorAll<HTMLInputElement>('[data-testid="pack-list"] input'),
    ];
  }
  function button(label: string): HTMLButtonElement {
    return [...target.querySelectorAll('button')].find((b) => b.textContent === label)!;
  }

  it('kind list renders a list editor: edit and remove rows emit string[]', () => {
    const onchange = vi.fn();
    render({
      field: field('proposal_denylist'),
      value: ['blurry_*', '*_scene'],
      onchange,
    });
    expect(target.querySelector('textarea')).toBeNull();
    expect(listInputs().map((i) => i.value)).toEqual(['blurry_*', '*_scene']);
    type(listInputs()[0]!, 'dark_*');
    expect(onchange).toHaveBeenLastCalledWith(['dark_*', '*_scene']);
    button('Remove').click();
    flushSync();
    expect(onchange).toHaveBeenLastCalledWith(['*_scene']);
  });

  it('the list shape follows the served kind, not the field id', () => {
    const base = field('proposal_denylist');
    render({ field: { ...base, field: 'other_list' }, value: ['a'] });
    expect(target.querySelector('[data-testid="pack-list"]')).not.toBeNull();
    unmount(instance!);
    instance = undefined;
    target.remove();
    render({ field: { ...base, kind: 'text' }, value: 'plain' });
    expect(target.querySelector('[data-testid="pack-list"]')).toBeNull();
    expect(target.querySelector('textarea')).not.toBeNull();
  });

  it('adding a row shows a blank input but never emits a blank entry', () => {
    const onchange = vi.fn();
    render({ field: field('proposal_denylist'), value: ['a'], onchange });
    button('Add pattern').click();
    flushSync();
    expect(listInputs()).toHaveLength(2);
    expect(onchange).not.toHaveBeenCalled();
    type(listInputs()[1]!, '  b_*  ');
    expect(onchange).toHaveBeenLastCalledWith(['a', 'b_*']);
  });

  it('blank and duplicate (case-insensitive) rows are dropped from the emitted list', () => {
    const onchange = vi.fn();
    render({ field: field('proposal_denylist'), value: ['a', 'b'], onchange });
    type(listInputs()[1]!, ' A ');
    expect(onchange).toHaveBeenLastCalledWith(['a']);
    expect(target.textContent).toContain('Duplicate');
    type(listInputs()[1]!, '   ');
    expect(onchange).toHaveBeenLastCalledWith(['a']);
  });

  it('limits: 200 chars per entry, add disabled at 500 entries', () => {
    const many = Array.from({ length: 500 }, (_, i) => `p${i}`);
    render({ field: field('proposal_denylist'), value: many });
    expect(listInputs()[0]!.maxLength).toBe(200);
    expect(button('Add pattern').disabled).toBe(true);
  });

  it('a missing value renders an empty list, never a string', () => {
    render({ field: field('proposal_denylist'), value: undefined });
    expect(target.textContent).toContain('No patterns.');
    expect(listInputs()).toHaveLength(0);
  });

  it('an unknown future kind degrades to a read-only view without crashing', () => {
    const onchange = vi.fn();
    render({
      field: { ...field('proposal_denylist'), kind: 'matrix' },
      value: { x: [1, 2] } as unknown as PackFieldValue,
      onchange,
    });
    expect(target.querySelector('[data-testid="pack-unknown"]')!.textContent).toContain(
      '"x"',
    );
    expect(target.querySelector('textarea')).toBeNull();
    expect(target.querySelectorAll('input')).toHaveLength(0);
    expect(onchange).not.toHaveBeenCalled();
  });

  it('proposal_denylist: the served issue on the field shows', () => {
    render({
      field: field('proposal_denylist'),
      value: [],
      issues: [
        issue({
          code: 'pack_field_missing',
          id: 'pack_field_missing',
          field: 'proposal_denylist',
          message: 'proposal_denylist must be a list of strings',
        }),
      ],
    });
    expect(target.textContent).toContain('proposal_denylist must be a list of strings');
  });

  it('read-only: no add/remove, inputs are read-only', () => {
    render({ field: field('synonyms'), value: { a: 'widget' }, readonly: true });
    expect(target.textContent).not.toContain('Add entry');
    expect(target.querySelector('input')!.readOnly).toBe(true);
  });

  it('shows only the issues it is given, as served', () => {
    render({
      field: field('class_user_template'),
      value: 'x',
      issues: [issue()],
    });
    const rows = target.querySelectorAll('[data-testid="config-issue"]');
    expect(rows).toHaveLength(1);
    expect(rows[0]!.getAttribute('data-code')).toBe('pack_placeholder_missing');
    expect(rows[0]!.textContent).toContain(
      'class_user_template must contain {class_names_csv}',
    );
  });
});

describe('ConfigIssueList', () => {
  it('renders severity, message, code, field on request and the served bypassable flag', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ConfigIssueList, {
      target,
      props: {
        showField: true,
        issues: [
          issue({ bypassable: true }),
          issue({
            code: 'pack_example_values',
            id: 'pack_example_values',
            severity: 'info',
            field: null,
            message: 'Examples: A1',
          }),
        ],
      },
    });
    flushSync();
    const rows = [...target.querySelectorAll('[data-testid="config-issue"]')];
    expect(rows.map((r) => r.getAttribute('data-severity'))).toEqual(['error', 'info']);
    expect(rows[0]!.textContent).toContain('class_user_template');
    expect(rows[0]!.textContent).toContain('(can be overridden)');
    expect(rows[1]!.textContent).not.toContain('(can be overridden)');
    expect(rows[1]!.textContent).toContain('Examples: A1');
  });

  it('renders nothing for no issues', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(ConfigIssueList, { target, props: { issues: [] } });
    flushSync();
    expect(target.querySelector('[data-testid="config-issues"]')).toBeNull();
  });
});
