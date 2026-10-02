/**
 * One profile field, mounted: the served label, help, default and range
 * render; each served `type` gets its control and emits the value the body
 * stores (a number, a boolean, a list, a tuple, the served `choice.id`);
 * the served `empty_choice` and a stored value the list lacks both show;
 * a row whose `applies_when` is off in the saved revision is dimmed with a
 * note but still editable; only the served issues handed to it show.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { issue } from '$lib/test/fixtures/promptPacks';
import {
  profileSchemaFixture,
  vocabularyFixture,
} from '$lib/test/fixtures/regionProfiles';
import { choiceList } from '$lib/profiles/profileFields';
import type { ValidationIssue } from '$lib/types_config';
import type {
  Choice,
  ProfileFieldType,
  ProfileFieldValue,
  ProfileSchemaField,
} from '$lib/types_profiles';
import ProfileFieldEditor from './ProfileFieldEditor.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function field(id: string): ProfileSchemaField {
  return profileSchemaFixture().fields.find((f) => f.field === id)!;
}

function render(props: {
  field: ProfileSchemaField;
  value: ProfileFieldValue | undefined;
  issues?: ValidationIssue[];
  choices?: Choice[] | null;
  applies?: boolean | null;
  readonly?: boolean;
}) {
  const onchange = vi.fn();
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ProfileFieldEditor, {
    target,
    props: { issues: [], choices: null, applies: null, onchange, ...props },
  });
  flushSync();
  return onchange;
}

function fire(el: Element, value: string, event = 'input') {
  (el as HTMLInputElement).value = value;
  el.dispatchEvent(new Event(event, { bubbles: true }));
  flushSync();
}

const q = <T extends Element = HTMLElement>(sel: string) => target.querySelector<T>(sel);

describe('ProfileFieldEditor', () => {
  it('renders the served label, help, default and range', () => {
    render({ field: field('max_regions_per_item'), value: 4 });
    expect(target.textContent).toContain('Most regions per item');
    expect(target.textContent).toContain('Machine proposals kept per item.');
    expect(q('[data-testid="field-facts"]')!.textContent).toMatch(
      /default: 1\s*· 1 to 64/,
    );
  });

  it('int: a number input that emits numbers, and null when emptied', () => {
    const onchange = render({ field: field('max_regions_per_item'), value: 4 });
    const input = q<HTMLInputElement>('input[type="number"]')!;
    expect(input.min).toBe('1');
    expect(input.max).toBe('64');
    fire(input, '8');
    expect(onchange).toHaveBeenLastCalledWith(8);
    fire(input, '');
    expect(onchange).toHaveBeenLastCalledWith(null);
  });

  it('string with choices_from: the served empty choice, choices and the stored value', () => {
    const choices = choiceList(vocabularyFixture(), 'detectors');
    const onchange = render({
      field: field('detector_model'),
      value: 'retired_v0',
      choices,
    });
    const select = q<HTMLSelectElement>('[data-testid="field-choice"]')!;
    const labels = [...select.options].map((o) => [o.value, o.textContent]);
    expect(labels).toEqual([
      ['', 'No detector leg'],
      ['tag_detector_v1', 'tag_detector_v1 (promoted)'],
      ['item_detector_base', 'item_detector_base'],
      ['retired_v0', 'retired_v0 (not in the list)'],
    ]);
    expect(select.value).toBe('retired_v0');
    fire(select, 'tag_detector_v1', 'change');
    expect(onchange).toHaveBeenLastCalledWith('tag_detector_v1');
    fire(select, '', 'change');
    expect(onchange).toHaveBeenLastCalledWith('');
  });

  it('enum: the served options', () => {
    const onchange = render({ field: field('text_reader'), value: 'none' });
    const select = q<HTMLSelectElement>('select')!;
    expect([...select.options].map((o) => o.textContent)).toEqual([
      'Off (region has no text)',
      'VLM',
      'OCR',
    ]);
    fire(select, 'ocr', 'change');
    expect(onchange).toHaveBeenLastCalledWith('ocr');
  });

  it('enum with choices_from and no static enum: the served list is the options', () => {
    const dynamic: ProfileSchemaField = {
      ...field('text_reader'),
      enum: null,
      choices_from: 'text_reader_modes',
    };
    const choices = choiceList(vocabularyFixture(), 'text_reader_modes');
    const onchange = render({ field: dynamic, value: 'none', choices });
    const select = q<HTMLSelectElement>('select')!;
    expect([...select.options].map((o) => o.value)).toEqual(choices!.map((c) => c.id));
    fire(select, 'ocr', 'change');
    expect(onchange).toHaveBeenLastCalledWith('ocr');
  });

  it('bool: a checkbox', () => {
    const onchange = render({ field: field('text_hint_enabled'), value: false });
    const box = q<HTMLInputElement>('input[type="checkbox"]')!;
    box.click();
    flushSync();
    expect(onchange).toHaveBeenLastCalledWith(true);
  });

  it('string_list with choices: chips by served label, remove, and add a name', () => {
    const choices = choiceList(vocabularyFixture(), 'registry_classes');
    const onchange = render({
      field: field('parent_classes'),
      value: ['widget'],
      choices,
    });
    expect(
      [...target.querySelectorAll('[data-testid="list-item"]')].map((c) =>
        c.textContent?.replace('×', '').trim(),
      ),
    ).toEqual(['widget']);
    const datalist = q('datalist')!;
    expect([...datalist.querySelectorAll('option')].map((o) => o.value)).toEqual([
      'gadget',
      'gizmo',
    ]);
    fire(q('[data-testid="list-add-input"]')!, 'gadget');
    q<HTMLButtonElement>('[data-testid="list-add"]')!.click();
    flushSync();
    expect(onchange).toHaveBeenLastCalledWith(['widget', 'gadget']);
    q<HTMLButtonElement>('button[aria-label="Remove widget"]')!.click();
    flushSync();
    expect(onchange).toHaveBeenLastCalledWith([]);
  });

  it('only a field the schema marks advanced carries the advanced badge', () => {
    render({ field: field('region_nms_iou'), value: 0.5 });
    expect(target.textContent).toContain('advanced');
    unmount(instance!);
    target.remove();
    render({ field: field('max_regions_per_item'), value: 4 });
    expect(target.textContent).not.toContain('advanced');
  });

  it('int_list: an added entry is stored as a number, and a non-number is refused', () => {
    const f = { ...field('max_regions_per_item'), type: 'int_list' as const };
    const onchange = render({ field: f, value: [1] });
    fire(q('[data-testid="list-add-input"]')!, 'abc');
    q<HTMLButtonElement>('[data-testid="list-add"]')!.click();
    flushSync();
    expect(onchange).not.toHaveBeenCalled();
    fire(q('[data-testid="list-add-input"]')!, '7');
    q<HTMLButtonElement>('[data-testid="list-add"]')!.click();
    flushSync();
    expect(onchange).toHaveBeenLastCalledWith([1, 7]);
  });

  it('rgb / float_pair: fixed-arity number inputs emit the whole tuple', () => {
    let onchange = render({ field: field('letterbox_fill'), value: [114, 114, 114] });
    const rgb = target.querySelectorAll<HTMLInputElement>(
      '[data-testid="field-tuple"] input',
    );
    expect(rgb).toHaveLength(3);
    fire(rgb[1]!, '0');
    expect(onchange).toHaveBeenLastCalledWith([114, 0, 114]);
    unmount(instance!);
    target.remove();

    onchange = render({ field: field('auto_confirm_area_frac'), value: [0.1, 0.9] });
    const pair = target.querySelectorAll<HTMLInputElement>(
      '[data-testid="field-tuple"] input',
    );
    expect(pair).toHaveLength(2);
    fire(pair[0]!, '0.2');
    expect(onchange).toHaveBeenLastCalledWith([0.2, 0.9]);
  });

  it('an unknown served type edits as JSON (a non-JSON value is sent raw)', () => {
    const onchange = render({
      field: { ...field('display_name'), type: 'box_map' as ProfileFieldType },
      value: { a: 1 },
    });
    const ta = q<HTMLTextAreaElement>('[data-testid="field-json"]')!;
    expect(ta.value).toBe('{"a":1}');
    fire(ta, '{"a":2}');
    expect(onchange).toHaveBeenLastCalledWith({ a: 2 });
    fire(ta, '{oops');
    expect(onchange).toHaveBeenLastCalledWith('{oops');
  });

  it('applies_when off in the saved revision: dimmed with a note, still editable', () => {
    render({ field: field('segmenter_text_prompt'), value: 'tag', applies: false });
    const row = q('[data-testid="profile-field"]')!;
    expect(row.classList.contains('opacity-60')).toBe(true);
    expect(q('[data-testid="field-not-applied"]')!.textContent).toContain(
      'segmenter is off',
    );
    expect(q<HTMLInputElement>('input')!.readOnly).toBe(false);
  });

  it('applies true or unknown: no note', () => {
    render({ field: field('segmenter_text_prompt'), value: 'tag', applies: true });
    expect(q('[data-testid="field-not-applied"]')).toBeNull();
  });

  it('read-only: no add or remove controls', () => {
    render({ field: field('parent_classes'), value: ['widget'], readonly: true });
    expect(q('[data-testid="list-add"]')).toBeNull();
    expect(q('button[aria-label="Remove widget"]')).toBeNull();
  });

  it('shows only the served issues handed to it', () => {
    render({
      field: field('max_regions_per_item'),
      value: 99,
      issues: [
        issue({
          code: 'profile_field_range',
          field: 'max_regions_per_item',
          message: 'max_regions_per_item must be 1 to 64',
        }),
      ],
    });
    const rows = target.querySelectorAll('[data-testid="config-issue"]');
    expect(rows).toHaveLength(1);
    expect(rows[0]!.textContent).toContain('must be 1 to 64');
  });
});
