/**
 * The combine class-mapping table, mounted: one row per served preview
 * class with its served `mapped_to`, a name input only for `create`, a
 * `map` picker limited to the names the form's own `create` rows define,
 * served action labels, and a touched marker once the operator edits.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import CombineMappingTable from './CombineMappingTable.svelte';
import { createCombineWizard } from '$lib/combine/combineWizardController.svelte';
import { combinePreview } from '$lib/test/fixtures/combine';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

function render() {
  const preview = combinePreview();
  // A served mapping that differs from the source class name, plus an unmapped row.
  preview.sources[1]!.classes = [
    { name: 'widget', count: 5, mapped_to: 'cog' },
    { name: 'tag', count: 1, mapped_to: null },
  ];
  const wizard = createCombineWizard({
    previewCombine: vi.fn().mockResolvedValue(preview),
    getDatasetFormatsFor: vi.fn().mockResolvedValue({
      mapping_actions: [
        { value: 'map', label: 'Map to class' },
        { value: 'create', label: 'Create class' },
        { value: 'skip', label: 'Skip' },
        { value: 'region', label: 'Region boxes' },
      ],
    }),
    projectOf: () => ({ prefix: '/p' }),
  });
  wizard.addSource('widgets-a');
  wizard.addSource('widgets-b');
  wizard.preview = preview;
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(CombineMappingTable, {
    target,
    props: { wizard, source: preview.sources[1]! },
  });
  flushSync();
  return { wizard, preview };
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const row = (name: string) =>
  target.querySelector(
    `[data-testid="combine-map-row"][data-class="${name}"]`,
  ) as HTMLElement;

function pick(el: HTMLSelectElement, value: string): void {
  el.value = value;
  el.dispatchEvent(new Event('change', { bubbles: true }));
  flushSync();
}

describe('CombineMappingTable', () => {
  it('renders one row per served class with its served mapping', () => {
    render();
    expect(target.querySelectorAll('[data-testid="combine-map-row"]')).toHaveLength(2);
    expect(
      row('widget')
        .querySelector('[data-testid="combine-map-served"]')
        ?.textContent?.trim(),
    ).toBe('cog');
    expect(
      row('tag').querySelector('[data-testid="combine-map-served"]')?.textContent?.trim(),
    ).toBe('unmapped');
    expect(row('widget').querySelector('td:nth-child(2)')?.textContent).toBe('5');
  });

  it('shows the target control that matches the chosen action', async () => {
    const { wizard } = render();
    expect(row('widget').querySelector('[data-testid="combine-map-name"]')).toBeNull();
    expect(row('widget').querySelector('[data-testid="combine-map-target"]')).toBeNull();

    pick(row('widget').querySelector('[data-testid="combine-map-action"]')!, 'create');
    expect(
      row('widget').querySelector('[data-testid="combine-map-name"]'),
    ).not.toBeNull();
    expect(wizard.choiceFor('widgets-b', 'widget')).toMatchObject({
      action: 'create',
      new_class_name: 'widget',
      touched: true,
    });
    expect(row('widget').getAttribute('data-touched')).toBe('true');

    pick(row('widget').querySelector('[data-testid="combine-map-action"]')!, 'skip');
    expect(row('widget').querySelector('[data-testid="combine-map-name"]')).toBeNull();
    expect(row('widget').querySelector('[data-testid="combine-map-target"]')).toBeNull();
  });

  it('the map picker lists only names defined by create rows, across sources', () => {
    const { wizard } = render();
    wizard.setChoice('widgets-a', 'gadget', { action: 'create', new_class_name: 'cog' });
    wizard.setChoice('widgets-a', 'widget', { action: 'skip' });
    pick(row('widget').querySelector('[data-testid="combine-map-action"]')!, 'map');
    const options = [
      ...row('widget').querySelectorAll('[data-testid="combine-map-target"] option'),
    ].map((o) => (o as HTMLOptionElement).value);
    expect(options).toEqual(['', 'cog']);
    pick(row('widget').querySelector('[data-testid="combine-map-target"]')!, 'cog');
    expect(wizard.choiceFor('widgets-b', 'widget')).toMatchObject({
      action: 'map',
      new_class_name: 'cog',
    });
  });

  it('labels actions from the served vocabulary once it has loaded', async () => {
    render();
    await vi.waitFor(() => {
      const opts = [
        ...row('widget').querySelectorAll('[data-testid="combine-map-action"] option'),
      ].map((o) => o.textContent);
      expect(opts).toContain('Region boxes');
    });
  });

  it('renders the preview issues that name this source', () => {
    const { wizard, preview } = render();
    wizard.preview = {
      ...preview,
      errors: [
        {
          code: 'unmapped_class',
          id: 'unmapped_class',
          project: 'widgets-b',
          message: 'class widget has no mapping',
        },
        {
          code: 'unmapped_class',
          id: 'unmapped_class',
          project: 'widgets-a',
          message: 'other source',
        },
      ],
    };
    flushSync();
    const text = target.textContent ?? '';
    expect(text).toContain('class widget has no mapping');
    expect(text).not.toContain('other source');
  });
});
